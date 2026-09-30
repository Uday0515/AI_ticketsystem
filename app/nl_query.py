from __future__ import annotations

import ast
import json
import re
from typing import Any

import pandas as pd

from .anomalies import detect_anomalies
from .llm import LLMClient, LLMError


class UnsafeExpression(ValueError):
    pass


class ExpressionValidator(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.Dict,
        ast.List,
        ast.Tuple,
        ast.Set,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Subscript,
        ast.Slice,
        ast.BinOp,
        ast.BoolOp,
        ast.Compare,
        ast.UnaryOp,
        ast.Attribute,
        ast.Call,
        ast.keyword,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Invert,
        ast.Eq,
        ast.NotEq,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.In,
        ast.NotIn,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.BitAnd,
        ast.BitOr,
        ast.USub,
    )
    allowed_names = {
        "df",
        "pd",
        "True",
        "False",
        "None",
        "int",
        "float",
        "str",
        "len",
    }
    allowed_methods = {
        "head",
        "tail",
        "sum",
        "mean",
        "count",
        "size",
        "to_dict",
        "sort_values",
        "groupby",
        "isin",
        "isna",
        "notna",
        "eq",
        "ne",
        "fillna",
        "round",
        "reset_index",
        "loc",
        "shape",
        "value_counts",
        "dt",
        "year",
        "month",
    }

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, self.allowed_nodes):
            raise UnsafeExpression(f"Disallowed syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            raise UnsafeExpression(f"Unknown name: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        allowed_root_attribute = isinstance(node.value, ast.Name) and node.value.id in {
            "df",
            "pd",
        }
        if node.attr.startswith("_") or (
            node.attr not in self.allowed_methods and not allowed_root_attribute
        ):
            raise UnsafeExpression(f"Disallowed attribute: {node.attr}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise UnsafeExpression("Only named calls are allowed")
        if isinstance(node.func, ast.Name) and node.func.id not in {
            "int",
            "float",
            "str",
            "len",
        }:
            raise UnsafeExpression(f"Disallowed function: {node.func.id}")
        self.generic_visit(node)


def validate_expression(expression: str) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    ExpressionValidator().visit(tree)
    return tree


def _serialize(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.head(100).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.head(100).to_dict()
    if hasattr(value, "item"):
        return value.item()
    return value


class NLQueryEngine:
    def __init__(self, frame: pd.DataFrame, client: LLMClient | None = None):
        self.frame = frame
        self.client = client or LLMClient()

    def answer(self, question: str) -> dict[str, Any]:
        if not question.strip() or len(question) > 500:
            raise ValueError("question must be 1-500 characters")
        if self._is_anomaly_question(question):
            days = 7 if "week" in question.lower() else None
            anomalies = detect_anomalies(self.frame, days=days)
            return {
                "question": question,
                "generated_code": None,
                "raw_result": anomalies,
                "answer": f"Found {len(anomalies)} anomalies.",
            }
        if self.client.provider == "mock":
            return self._answer_locally(question)
        prompt = self._prompt(question)
        expression = self._generate(prompt)
        raw = self._evaluate(expression)
        serialized = _serialize(raw)
        answer = self.client.complete(self._answer_messages(serialized))
        return {
            "question": question,
            "generated_code": expression,
            "raw_result": serialized,
            "answer": answer,
        }

    def _answer_locally(self, question: str) -> dict[str, Any]:
        """Answer common CSV questions without silently inventing a result."""
        text = question.lower()
        frame = self.frame
        column = self._find_column(text)
        filtered = self._filter_frame(frame, text)

        if any(word in text for word in ("how many", "count", "number of")) or (
            column is not None and any(word in text for word in ("most", "top", "highest"))
        ):
            result: Any = {"count": int(len(filtered))}
            if column and column != "ticket_id" and any(
                word in text
                for word in ("by", "each", "per", "most", "top", "highest")
            ):
                counts = filtered[column].value_counts(dropna=False)
                result = counts.head(1).to_dict() if any(
                    word in text for word in ("most", "highest", "top")
                ) else counts.to_dict()
        elif any(word in text for word in ("average", "avg", "mean")):
            numeric = self._numeric_column(text)
            if numeric is None:
                raise ValueError("Specify a numeric CSV column, such as response time or customer rating")
            result = {"average": float(filtered[numeric].mean())}
        elif "sum" in text or "total" in text:
            numeric = self._numeric_column(text)
            if numeric is None:
                raise ValueError("Specify a numeric CSV column to total")
            result = {"total": float(filtered[numeric].sum())}
        elif any(word in text for word in ("maximum", "max", "highest", "longest")):
            numeric = self._numeric_column(text)
            if numeric is None:
                raise ValueError("Specify a numeric CSV column")
            result = {"maximum": float(filtered[numeric].max())}
        elif any(word in text for word in ("minimum", "min", "lowest", "shortest")):
            numeric = self._numeric_column(text)
            if numeric is None:
                raise ValueError("Specify a numeric CSV column")
            result = {"minimum": float(filtered[numeric].min())}
        elif any(word in text for word in ("list", "show", "what are")) or (
            "which" in text and "most" not in text and "highest" not in text
        ):
            result = _serialize(filtered.head(100))
        else:
            raise ValueError(
                "I could not map that question to the CSV. Ask about a column, count, "
                "average, total, minimum, maximum, list, or group."
            )

        serialized = _serialize(result)
        return {
            "question": question,
            "generated_code": None,
            "raw_result": serialized,
            "answer": self._local_answer_text(serialized),
        }

    def _find_column(self, text: str) -> str | None:
        aliases = {
            "created": "created_at",
            "category": "category",
            "priority": "priority",
            "status": "status",
            "response": "response_time_hrs",
            "resolution": "resolution_time_hrs",
            "agent": "agent_id",
            "rating": "customer_rating",
            "issue": "issue_summary",
            "age": "age_hrs",
            "ticket": "ticket_id",
        }
        for alias, name in aliases.items():
            if alias in text:
                return name
        return None

    def _numeric_column(self, text: str) -> str | None:
        column = self._find_column(text)
        if column in {"response_time_hrs", "resolution_time_hrs", "customer_rating", "age_hrs"}:
            return column
        return None

    def _filter_frame(self, frame: pd.DataFrame, text: str) -> pd.DataFrame:
        result = frame
        for column in ("category", "priority", "status"):
            for value in result[column].dropna().unique():
                if str(value).lower() in text:
                    result = result[result[column] == value]
        if "unresolved" in text or "not resolved" in text:
            result = result[~result["is_resolved"]]
        elif "resolved" in text:
            result = result[result["is_resolved"]]
        reference_time = frame.attrs.get("reference_time")
        if reference_time is not None:
            reference_time = pd.Timestamp(reference_time)
            if "this month" in text or "current month" in text:
                result = result[
                    (result["created_at"].dt.year == reference_time.year)
                    & (result["created_at"].dt.month == reference_time.month)
                ]
            elif "this year" in text or "current year" in text:
                result = result[result["created_at"].dt.year == reference_time.year]
            else:
                match = re.search(r"(?:last|past)\s+(\d+)\s+days?", text)
                if match:
                    cutoff = reference_time - pd.Timedelta(days=int(match.group(1)))
                    result = result[result["created_at"] >= cutoff]
        return result

    @staticmethod
    def _local_answer_text(result: Any) -> str:
        if isinstance(result, dict) and set(result) == {"count"}:
            return f"There are {result['count']} matching tickets."
        return f"Here is the result: {json.dumps(result, default=str)}"

    def _evaluate(self, expression: str) -> Any:
        try:
            tree = validate_expression(expression)
            return eval(
                compile(tree, "<generated>", "eval"),
                {"__builtins__": {}},
                {
                    "df": self.frame,
                    "pd": pd,
                    "int": int,
                    "float": float,
                    "str": str,
                    "len": len,
                },
            )
        except Exception as exc:
            raise ValueError(f"Generated expression failed: {exc}") from exc

    @staticmethod
    def _answer_messages(result: Any) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": "Answer briefly from the supplied result."},
            {"role": "user", "content": json.dumps(result, default=str)},
        ]

    def _generate(self, prompt: str) -> str:
        payload = json.loads(
            self.client.complete([{"role": "system", "content": prompt}])
        )
        expression = payload.get("expression")
        if not isinstance(expression, str):
            raise LLMError("LLM response did not contain an expression")
        return expression

    def _is_anomaly_question(self, question: str) -> bool:
        return "anomal" in question.lower() or "outlier" in question.lower()

    def _prompt(self, question: str) -> str:
        sample = self.frame.head(3).to_json(orient="records", date_format="iso")
        return f"""
Generate one pandas expression using df and return it as JSON.
Do not include markdown.

Question: {question}
Reference time: {self.frame.attrs.get('reference_time')}
Columns: {list(self.frame.columns)}
Sample rows: {sample}

Example for counting open tickets:
{{"expression": "{{'count': int((df['status'] == 'Open').sum())}}"}}
"""
