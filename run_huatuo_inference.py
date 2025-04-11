import sglang as sgl
from transformers import AutoTokenizer
import os

def main():
    # Get model path from config
    model_path = "FreedomIntelligence/HuatuoGPT-o1-8B"
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    
    # Load model
    print("Loading model...")
    engine = sgl.Engine(model_path=model_path)
    
    # Question
    question = """A 72-year-old man with a history of symptomatic gallstones presents with acute right-sided abdominal pain, right-upper-quadrant tenderness, and leukocytosis. Initial management includes antibiotics and hydration, but the clinical picture does not improve over three days. A barium-enema examination reveals partial obstruction at the level of the right transverse colon with a segmental area of narrowing at the distal limb of the hepatic flexure. The mucosal folds are nodular and thickened but not ulcerated. Glucagon administration does not relieve the narrowing. The patient passes large volumes of liquid stools that are negative for occult blood. At operation, considerable distention of the small bowel is found with an area of apparent necrosis on its antimesenteric border adjacent to a pelvic abscess. The ascending colon is distended, and a mass in the right transverse colon encompasses the fundus of the gallbladder and is wrapped in omentum. A right colectomy is performed, removing a segment of small bowel from a point just proximal to the sealed perforation in the pelvis, with an end-to-end anastomosis of the ileum to the midtransverse colon. The fundus of the gallbladder is amputated with the specimen, and a cholecystotomy tube is inserted into its stump. Postoperatively, the patient initially improves but is readmitted two months later with fever, right-upper-quadrant pain, and elevated bilirubin. Ultrasound suggests a common-duct stone, but retrograde cholangiography reveals stenosis of the distal centimeter of the duct, treated by endoscopic papillotomy. What is the most likely etiology of the patient's readmission symptoms?"""
    
    # Format using chat template
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Generate response
    print("Generating response...")
    response = engine.generate(
        prompt=prompt,
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 8192,
            "top_p": 0.9
        }
    )
    
    # Save response
    print("Saving response...")
    with open("huatuo_response.txt", "w") as f:
        f.write(f"Question:\n{question}\n\nResponse:\n{response['text']}")
    
    print("Done! Response saved to huatuo_response.txt")

if __name__ == "__main__":
    main()