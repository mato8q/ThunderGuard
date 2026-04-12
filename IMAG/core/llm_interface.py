import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class TargetLLM:
    def __init__(self, model_name_or_path, device="cuda"):
        self.device = device
        print(f"Loading model {model_name_or_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def generate(self, prompt, max_new_tokens=256):
        """สร้างข้อความตอบกลับจาก prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def extract_hidden_states(self, prompt, target_layer=-1):
        """
        ดึงค่า Hidden state ของคำสุดท้าย (Last token) จากเลเยอร์ที่ระบุ
        target_layer: ใช้ -1 สำหรับเลเยอร์สุดท้ายเสมอ (ป้องกัน Error กรณีเปลี่ยนโมเดล)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            all_hidden_states = outputs.hidden_states

            if all_hidden_states is None:
                raise ValueError("โมเดลไม่ส่ง hidden_states กลับมา ตรวจสอบการตั้งค่าโมเดล")

            layer_hidden_states = all_hidden_states[target_layer]
            last_token_hidden_state = layer_hidden_states[:, -1, :].to(torch.float32)

        return last_token_hidden_state.cpu().numpy()


if __name__ == "__main__":
    llm = TargetLLM("Qwen/Qwen1.5-0.5B-Chat")
    prompt = "How to make a bomb?"
    h_x = llm.extract_hidden_states(prompt, target_layer=-1)
    print(f"Shape ของ Activation Vector: {h_x.shape}")
