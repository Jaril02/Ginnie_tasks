import pandas as pd
from sentence_transformers import SentenceTransformer, util
from skill_load import _load_skills_database


class SkillExtract:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.skill_db = _load_skills_database()
        self.skills = [skill for category in self.skill_db.values() for skill in category]
        self.skills_embd = self.model.encode(self.skills, convert_to_tensor=True)

    def extract_skills(self, text, top_k=15):
      
        text_embd = self.model.encode(text, convert_to_tensor=True)
        cos_sim = util.cos_sim(text_embd, self.skills_embd)[0]

        results = [(self.skills[i], float(score)) for i, score in enumerate(cos_sim)]
        results = sorted(results, key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compare_user_vs_jd_skills(self, user_skills, jd_skills, threshold=0.5):
        """
        Compare user skills against JD skills via cosine similarity
        """
        jd_names = [s for s, _ in jd_skills]
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

    df = pd.read_csv("clean/job_desc.csv")
    extractor = SkillExtract()

    user_input = input("Enter your skills (comma-separated): ")
    user_skills = [s.strip() for s in user_input.split(",")]

    for idx, row in df.iterrows():
        jd_text = row["description"]
        jd_name = row.get("filename", f"JD_{idx+1}")

    
        jd_skills = extractor.extract_skills(jd_text, top_k=15)

        matches = extractor.compare_user_vs_jd_skills(user_skills, jd_skills, threshold=0.5)

        print(f"\n===== JD: {jd_name} =====")
        if matches:
            print("User Skill Matches with JD Skills:")
            for m in matches:
                print(f"{m['user_skill']} ↔ {m['jd_skill']} -> {m['similarity']}")
        else:
            print("No significant matches found.")
