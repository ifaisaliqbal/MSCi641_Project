from bert_serving.client import BertClient
bc = BertClient()
temp  = bc.encode(['First do it'])
print(temp)