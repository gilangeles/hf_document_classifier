# HuggingFace-based Document Classifier

## How to install the package:
run in your terminal:
```
pip install git+https://github.com/gilangeles/hf_document_classifier
```
## How to use the package
Use the code below as a template for using the package

```
from hf_document_classifier.classifier import HFDocumentClassifier

classifier = HFDocumentClassifier("/my/model/dir")

classifier.classify_text(
    [
        "my text here",
        "next text here",
        "so on and so forth"
    ]
)
```
This is for abstracting the instantiation and usage of __HuggingFace__ based document classifier.
