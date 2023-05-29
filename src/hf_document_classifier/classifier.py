import typing

import more_itertools
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    pipeline,
)


class HFDocumentClassifier:
    def __init__(
        self, model_path: str, framework: str = "tf", device: typing.Optional[int] = 0
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if framework == "tf":
            self.model = TFAutoModelForSequenceClassification.from_pretrained(
                model_path
            )
        elif framework == "pt":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path, low_cpu_mem_usage=True
            )

        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )

    def classify_text(
        self, texts: typing.List[str], batchsize: int = 3
    ) -> typing.List[typing.Dict]:
        results = []
        batched_text = more_itertools.batched(texts, batchsize)
        for text in batched_text:
            results.extend(self.classifier(text))
        return results
