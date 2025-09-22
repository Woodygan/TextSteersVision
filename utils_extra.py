import re
import string


# Official ChartQA evaluation metrics from lm-evaluation-harness
# Source: https://github.com/EleutherAI/lm-evaluation-harness

def _normalize_string(s):
    """Remove surrounding quotes from string."""
    if (s.startswith('"') and s.endswith('"')) or (
        s.startswith("'") and s.endswith("'")
    ):
        return s[1:-1]
    return s


def _remove_end_punctuation(unnormalized_string: str) -> str:
    """Remove trailing punctuation except %."""
    while (
        unnormalized_string
        and (
            unnormalized_string[-1] in string.punctuation
            or unnormalized_string[-1].isspace()
        )
        and unnormalized_string[-1] != "%"
    ):
        unnormalized_string = unnormalized_string[:-1]
    return unnormalized_string


class RelaxedCorrectness:
    """Official ChartQA relaxed correctness metrics.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    "Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct."
    """

    def _relaxed_correctness(
        self, prediction: str, targets: list[str], max_relative_change: float = 0.05
    ) -> float:
        def _to_float(text: str) -> tuple[float | None, bool]:
            text = text.strip()
            is_percent = text.endswith("%")
            try:
                value = float(text.rstrip("%"))
                return value, is_percent
            except ValueError:
                return None, False

        def _is_letter(text: str) -> bool:
            return text.isalpha() and len(text) == 1

        def _preprocess_text(text: str) -> str:
            if not any(char.isdigit() for char in text):
                return _normalize_string(text)
            else:
                return _remove_end_punctuation(text).replace(",", "").replace("$", "")

        def calculate_relative_change(prediction: float, target: float) -> float:
            return abs(prediction - target) / max(abs(target), 1e-10)

        def _compare_numeric_values(
            prediction: float, target: float, max_relative_change: float
        ) -> float:
            relative_change = calculate_relative_change(prediction, target)
            return 1.0 if relative_change <= max_relative_change else 0.0

        def _compare_text_values(prediction: str, target: str) -> float:
            while prediction and prediction[-1] in string.punctuation:
                prediction = prediction[:-1]
            return 1.0 if prediction.lower() == target.lower() else 0.0

        def _to_decimal(value: float, is_percent: bool) -> float:
            return value / 100 if is_percent else value

        def _compare_numeric_with_percent(
            prediction: float,
            prediction_is_percent: bool,
            target: float,
            target_is_percent: bool,
            max_relative_change: float,
        ) -> float:
            # Compare as-is
            value = _compare_numeric_values(prediction, target, max_relative_change)

            # If not equal and one is percent, try other comparisons
            if value != 1.0 and (prediction_is_percent or target_is_percent):
                value = max(
                    value,
                    _compare_numeric_values(
                        _to_decimal(prediction, prediction_is_percent),
                        target,
                        max_relative_change,
                    ),
                    _compare_numeric_values(
                        prediction,
                        _to_decimal(target, target_is_percent),
                        max_relative_change,
                    ),
                )
            return value

        prediction = _preprocess_text(prediction)
        prediction_float, prediction_is_percent = _to_float(prediction)

        value_list = []
        for target in targets:
            target = _preprocess_text(target)
            target_float, target_is_percent = _to_float(target)

            if prediction_float is not None and target_float is not None:
                # Compare as numeric values
                value = _compare_numeric_with_percent(
                    prediction_float,
                    prediction_is_percent,
                    target_float,
                    target_is_percent,
                    max_relative_change,
                )
            elif _is_letter(target) and len(prediction) > 0:
                # Compare as multiple choice options: take first letter from prediction
                value = 1.0 if prediction[0].lower() == target.lower() else 0.0
            else:
                # Compare as text values
                value = _compare_text_values(prediction, target)

            value_list.append(value)

        return max(value_list)

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        reference_answer = (
            reference_answer
            if isinstance(reference_answer, list)
            else [reference_answer]
        )
        return self._relaxed_correctness(model_answer, reference_answer)


class ExplicitPromptRelaxedCorrectness(RelaxedCorrectness):
    """Relaxed correctness for explicit prompt with answer extraction."""

    @property
    def name(self) -> str:
        return "explicit_prompt_relaxed_correctness"

    def _get_final_answer(self, generation: str) -> str:
        def _find_last_occurrence(pattern: str, string: str):
            return string.rfind(pattern)

        # Strip extraneous markdown around the answer:
        generation = re.sub(r"([aA]nswer)\**:\**", "\\1:", generation)

        final_answer_index = _find_last_occurrence("answer:", generation.lower())

        if final_answer_index != -1:
            # Find the start of the answer (after "final answer:")
            start_index = final_answer_index + len("answer:")

            # Split the remaining text into lines
            lines = generation[start_index:].split("\n")

            # Find the first non-empty line
            final_answer = next((line.strip() for line in lines if line.strip()), "")

            # Remove any markdown formatting
            final_answer = re.sub(r"[*_\[\]\(\)]", "", final_answer)

            return final_answer
        else:
            return ""

    def score(self, model_answer: str, reference_answer: str | list[str]) -> float:
        parsed_model_answer = self._get_final_answer(model_answer)
        if not parsed_model_answer:
            # Parsing failed.
            return 0.0
        return super().score(parsed_model_answer, reference_answer)


def normalize_chartqa_answer_official(predicted: str, target: str) -> bool:
    """
    Official ChartQA answer normalization using the exact implementation
    from lm-evaluation-harness.
    
    Args:
        predicted: Model's predicted answer
        target: Ground truth answer
        
    Returns:
        Boolean indicating if the prediction is correct
    """
    evaluator = RelaxedCorrectness()
    score = evaluator.score(predicted, target)
    return score == 1.0


def relaxed_accuracy_official(references, predictions):
    """
    Official relaxed accuracy function from lm-evaluation-harness.
    Used for evaluation with explicit answer prompts.
    """
    pred = predictions
    ref = references
    score = ExplicitPromptRelaxedCorrectness().score(pred, ref)
    return score == 1.0