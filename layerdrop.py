def drop_layers(model, layer2drop=[]):
    cls = type(model.transformer.layer)
    model.transformer.layer = cls(lay for i, lay in enumerate(model.transformer.layer) \
                                                                if i not in layer2drop)
    return model

if __name__ == "__main__":
    import torch
    from transformers import DistilBertTokenizer, DistilBertModel
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    print(model)
    print('-------')
    print(drop_layers(model, [2, 5]))
