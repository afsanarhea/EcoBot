import json
import os
from datetime import datetime
from ecobot_rag import EcobotRAG

with open("golden_dataset.json", "r") as f:
    golden_dataset = json.load(f)

print("  ECOBOT - 5 TOOL EVALUATION")
print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  Questions: {len(golden_dataset)}")

print("\nLoading EcoBot")
bot = EcobotRAG()
bot.load_vector_store()
bot.setup_qa_chain()
print("EcoBot ready!\n")

results = []

phase_a_ids = [1, 8, 14, 20, 28]
phase_a_questions = [q for q in golden_dataset if q["id"] in phase_a_ids]

for q in phase_a_questions:
    print(f"\n--- Q{q['id']} [{q['category']}] [{q['plant']}] ---")
    print(f"Q: {q['question']}")
    
    try:
        response = bot.query(q["question"])
        answer = response["result"]
    except Exception as e:
        answer = f"ERROR: {str(e)}"
    
    print(f"A: {answer[:150]}...")
    
    found = []
    missing = []
    for kw in q["expected_keywords"]:
        if kw.lower() in answer.lower():
            found.append(kw)
        else:
            missing.append(kw)
    
    keyword_score = len(found) / len(q["expected_keywords"]) * 10
    
    print(f"Keywords: {len(found)}/{len(q['expected_keywords'])} matched | Score: {keyword_score:.1f}/10")
    if missing:
        print(f"Missing: {missing}")
    
    results.append({
        "id": q["id"],
        "question": q["question"],
        "expected": q["expected_answer"],
        "actual_answer": answer,
        "category": q["category"],
        "plant": q["plant"],
        "keywords_found": found,
        "keywords_missing": missing,
        "keyword_score": round(keyword_score, 1),
        "timestamp": datetime.now().isoformat()
    })

print("  EVALUATION SUMMARY")

categories = {}
for r in results:
    cat = r["category"]
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(r["keyword_score"])

print(f"\n{'Category':<25} {'Avg Score':<12} {'Questions'}")
print("-"*50)
for cat, scores in categories.items():
    avg = sum(scores) / len(scores)
    print(f"{cat:<25} {avg:.1f}/10      {len(scores)}")

all_scores = [r["keyword_score"] for r in results]
overall = sum(all_scores) / len(all_scores)
print(f"\n{'OVERALL':<25} {overall:.1f}/10      {len(all_scores)}")

print("  POTENTIAL ISSUES")
for r in results:
    if r["keyword_score"] < 5:
        print(f"\n Q{r['id']}: {r['question'][:60]}...")
        print(f"   Score: {r['keyword_score']}/10")
        print(f"   Missing: {r['keywords_missing']}")

print("  DEEPEVAL EVALUATION")

try:
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from deepeval import evaluate as deepeval_evaluate

    test_cases = []
    
    for r in results:
        test_cases.append(
            LLMTestCase(
                input=r["question"],
                actual_output=r["actual_answer"],
                expected_output=r["expected"],
                retrieval_context=[r["expected"]],
            )
        )
    
    relevancy = GEval(
        name="Relevancy",
        criteria="Does the response directly and specifically address the user's question about plant care?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gemini/gemini-2.5-flash",
    )
    
    faithfulness = GEval(
        name="Faithfulness",
        criteria="Is the response grounded in the expected output without adding fabricated or hallucinated information?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model="gemini/gemini-2.5-flash",
    )
    
    coherence = GEval(
        name="Coherence",
        criteria="Is the response logically structured, clear, and easy to understand with proper formatting?",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gemini/gemini-2.5-flash",
    )
    
    toxicity = GEval(
        name="Toxicity",
        criteria="Is the response free from offensive, rude, or inappropriate language? Score 1 if clean, 0 if toxic.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gemini/gemini-2.5-flash",
    )

    print("\nRunning DeepEval metrics (this may take a few minutes)...\n")
    deepeval_evaluate(
        test_cases=test_cases,
        metrics=[relevancy, faithfulness, coherence, toxicity],
    )
    
except Exception as e:
    print(f"\nDeepEval evaluation skipped: {e}")
    print("Note: DeepEval GEval requires OpenAI API key (gpt-4o).")
    print("If you don't have OpenAI key, keyword-based evaluation above is still valid.")

today = datetime.now().strftime("%Y-%m-%d")
filename = f"results_{today}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

