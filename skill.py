import pandas as pd
from sentence_transformers import SentenceTransformer, util
from skill_load import _load_skills_database
from exp_ext import extract_experience

import nltk
nltk.download('punkt')

class SkillExtract:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.skill_db = _load_skills_database()
        self.skills = [skill for category in self.skill_db.values() for skill in category]
        self.skills_embd = self.model.encode(self.skills, convert_to_tensor=True)

    def extract_skills(self, text, threshold=0.65, top_k=15):
       
        found_skills = set()

       
        for skill in self.skills:
            if skill in text:
                found_skills.add(skill)

       
        sentences = nltk.sent_tokenize(text)
        for sent in sentences:
            sent_emb = self.model.encode(sent, convert_to_tensor=True)
            cos_sim = util.cos_sim(sent_emb, self.skills_embd)[0]

            for i, score in enumerate(cos_sim):
                if score >= threshold:
                    skill = self.skills[i]
       
                    if skill not in found_skills and any(word in sent for word in skill.split()):
                        found_skills.add(skill)

        return list(found_skills)
    @staticmethod
    def is_experience_match(jd_exp, user_exp):
        if jd_exp is None:
            return True 
        if jd_exp == 0:
            return user_exp == 0
        if isinstance(jd_exp, tuple):
            start, end = jd_exp
            return start <= user_exp <= end
        return False
    

    def compare_user_vs_jd_skills(self, user_skills, jd_skills, threshold=0.5):
        jd_names = jd_skills
        jd_embd = self.model.encode(jd_names, convert_to_tensor=True)
        user_embd = self.model.encode(user_skills, convert_to_tensor=True)

        matches = []
        for i, u_skill in enumerate(user_skills):
            cos_sim = util.cos_sim(user_embd[i], jd_embd)[0]
            for j, jd_skill in enumerate(jd_names):
                sim = round(cos_sim[j].item(), 3)
                if sim >= threshold:
                    matches.append({
                        "user_skill": u_skill,
                        "jd_skill": jd_skill,
                        "similarity": sim
                    })

        matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)
        return matches


if __name__ == "__main__":
 
    jd_df = pd.read_csv("clean/job_desc.csv")

    extractor = SkillExtract()
    user_experience = int(input("Enter your years of experience: "))
    # user_skills = ["python", "sql", "django", "aws"]
    user_input = input("Enter your skills (comma-separated): ").lower()
    user_skills = [s.strip() for s in user_input.split(",")]  

    similarities = []
    matched_pairs = []
    jd_experiences = []

    for idx, row in jd_df.iterrows():
        jd_text = row["description"]
        jd_name = row.get("filename", f"JD_{idx+1}")

        # Extract JD skills
        jd_skills = extractor.extract_skills(jd_text, top_k=15)   
        matches = extractor.compare_user_vs_jd_skills(user_skills, jd_skills, threshold=0.5)
        avg_score = sum(m["similarity"] for m in matches) / len(matches) if matches else 0

        similarities.append(avg_score)

        # formatted = "\n".join([f"{m['user_skill']} ↔ {m['jd_skill']} ({m['similarity']})" for m in matches])
        matched_pairs.append(matches)

        jd_exp = extract_experience(jd_text)
        jd_experiences.append(jd_exp)

    jd_df["jd_skills"] = jd_df["description"].apply(lambda x: [extractor.extract_skills(str(x), top_k=15)])
    jd_df["similarity"] = similarities
    jd_df["user_skills"] = ", ".join(user_skills)
    jd_df["matched_pairs"] = matched_pairs
    jd_df["Experience"]=jd_experiences
    jd_df["Eligible"] = jd_df["Experience"].apply(
        lambda x: "Positive" if extractor.is_experience_match(x, user_experience) else "Negative"
    )
    outfile = "output/desc_skill2.csv"
    


    jd_df.to_csv(outfile, index=False, encoding="utf-8")
    print(f"✅ Results saved to {outfile}")

   
    top_matches = jd_df.sort_values("similarity", ascending=False).head(5)
    print(top_matches[["filename", "jd_skills", "user_skills", "similarity","Eligible"]])
