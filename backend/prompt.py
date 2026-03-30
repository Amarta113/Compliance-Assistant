from langchain_core.prompts import ChatPromptTemplate

compliance_prompt = ChatPromptTemplate.from_template("""
        You are a GA4GH compliance expert reviewing a researcher's Data Use Letter.

        Compare the document against the GA4GH policy clauses below and identify gaps.

        RESEARCHER'S DOCUMENT:
        {research_text}

        GA4GH POLICY CLAUSES:
        {context}

        INSTRUCTIONS:
        - Every finding MUST cite the exact clause using [Source: policy_name | section | clause]
        - Flag gaps where GA4GH requirements are missing or incomplete
        - Flag compliant areas where the document aligns well

        Return ONLY valid JSON:
        [
        {{
            "type": "critical_gap" | "partial_gap" | "compliant",
            "topic": "e.g. Informed Consent",
            "finding": "what you found [Source: GA4GH Framework | Principle 4]",
            "gap": "what is missing",
            "recommendation": "specific fix needed",
            "citations": ["GA4GH Framework | Section 3 | Clause 3.2"],
            "confidence": 85
        }}
        ] """
        )