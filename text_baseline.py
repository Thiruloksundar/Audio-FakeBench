import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM


text = []
f = open('/home/thirulok/AV_DeepFake/val_metadata.json')
d = json.load(f)
for i in range(len(d)):
    if d[i]['modify_type']=='real' or d[i]['modify_type']=='audio_modified' or d[i]['modify_type']=='both_modified':
        f = open('/home/thirulok/AV_DeepFake/val_metadata/' + d[i]['file'][:-4]+'.json')
        data = json.load(f)
        transcript = ''
        print(i)
        for i in range(len(data['transcripts'])):
            s = data['transcripts'][i]['word'] + ' '
            transcript += s
        #text.append(transcript)

## v2 models
model_path = 'openlm-research/open_llama_3b_v2'
# model_path = 'openlm-research/open_llama_7b_v2'

## v1 models
# model_path = 'openlm-research/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'
# model_path = 'openlm-research/open_llama_13b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map='auto',
)

prompt = 'Q: ' + transcript + 'Is the given transcript real?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

generation_output = model.generate(
    input_ids=input_ids, max_new_tokens=32
)
print(tokenizer.decode(generation_output[0]))

'''import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load pre-trained model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Function to perform inference
def predict(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted label
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # Map the predicted label to the corresponding class
    label_map = {0: 'Real', 1: 'Manipulated'}
    return label_map[predicted_class]

# Example usage
texts = [transcript]
preds = []
for t in text:
    prediction = predict(t)
    preds.append(prediction)
    #print(f"Text: {t}\nPrediction: {prediction}\n")
a=0
for i in range(len(preds)):
    if preds[i]=='Real' and d[i]['modify_type']=='real':
        a+=1
    elif preds[i]=='Fake' and d[i]['modify_type']!='real':
        a+=1
print(a)'''
