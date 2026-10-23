"""Independent supervised groups retain their own document-mean denominator."""
from rlstack.training.post.base import PostProcessor, postprocessor


@postprocessor("sft_document_weights")
class SftDocumentWeights(PostProcessor):
    produces = ("sft_document_weight",)

    async def process(self, group, data, client):
        if not len(group):
            raise ValueError("an independent supervised group cannot be empty")
        return {"sft_document_weight": [1.0 / len(group)] * len(group)}
