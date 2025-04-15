from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
import torch

class LLMClassifier():
    def __init__(self): # Do not change the model unless you are very sure it works with the same prompt structure!!!
        torch.set_default_device("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moderation_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
        self.moderation_labels = ['needs moderation', 'acceptable']
        
        model_name = "mariagrandury/roberta-base-finetuned-sms-spam-detection"
        self.spam_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.spam_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    #TODO add support for batched reviews for efficiency
    def classify_moderation(self, review):
        result = self.moderation_classifier(review, candidate_labels = self.moderation_labels)
        top_label = result['labels'][0]
        confidence = result['scores'][0]
        return top_label, confidence

    def isSpam(self,text):
        inputs = self.spam_tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.spam_model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()

        # Map model output to custom categories
        return predicted_label == 1
    


if __name__ == '__main__':
    llm = LLMClassifier()
    pos_rev = "I visited this amusement park and it was a blast – the rides were thrilling and the atmosphere was festive. It was a great day out."
    neg_rev = "I visited this shitty amusement park and it was fucking awful – the rides were thrilling if you are a five year old. I want a refund... NOW!"
    spam_rev = "CONGRATULATIONS! Wanna get rich quick? You won a new IPhone! Visit my website for more information about my prices, special discounts are available here: https://andreacacioli.netlify.app/"
    for r in [pos_rev, neg_rev, spam_rev]:
        print(r)
        print(llm.classify_moderation(r))
        print(llm.isSpam(r))
        print()
