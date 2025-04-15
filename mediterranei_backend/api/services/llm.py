from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class LLM():
    def __init__(self, model_name = "microsoft/phi-2"): # Do not change the model unless you are very sure it works with the same prompt structure!!!
        torch.set_default_device("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)



    def ask(self, prompt) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = self.model.generate(**inputs, max_length=200)
        text = self.tokenizer.batch_decode(outputs)[0]
        return text

if __name__ == '__main__':
    llm = LLM()
    pos_rev = "I visited this amusement park and it was a blast – the rides were thrilling and the atmosphere was festive. It was a great day out."
    neg_rev = "I visited this shitty amusement park and it was fucking awful – the rides were thrilling if you are a five year old. I want a refund... NOW!"
    spam_rev = "CONGRATUKATIONS! You won a new IPhone! Visit my website for more information about my prices, special discounts are available here: https://andreacacioli.netlify.app/"

    print(llm.ask(f"Instruct: Out of these 3 classes: (Appropriate, Inappropriate, SPAM) [ONLY PICK ONE CLASS AND ONLY WRITE A SINGLE WORD, DO NOT TELL ANYTHING ELSE], considering the following review: \n {pos_rev} \n how would a human moderator consider this? \n Output: "))