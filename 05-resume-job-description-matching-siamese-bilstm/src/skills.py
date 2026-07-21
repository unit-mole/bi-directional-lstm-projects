from __future__ import annotations

# A deliberately small, transparent catalog for portfolio explainability.
# Production systems should use a governed taxonomy and domain review.
SKILL_CATALOG: dict[str, tuple[str, ...]] = {
    "Python": ("python", "pandas", "numpy", "flask", "django"),
    "SQL": ("sql", "database", "query", "queries"),
    "Machine Learning": ("machine learning", "ml", "modeling", "predictive modeling"),
    "Deep Learning": ("deep learning", "tensorflow", "keras", "pytorch", "neural network"),
    "NLP": ("nlp", "natural language processing", "text analytics", "language model"),
    "Statistics": ("statistics", "statistical", "hypothesis testing", "experimentation"),
    "Data Visualization": ("power bi", "tableau", "dashboard", "dashboarding", "visualization"),
    "Cloud": ("aws", "azure", "gcp", "cloud"),
    "Docker": ("docker", "container", "containerization"),
    "Kubernetes": ("kubernetes", "k8s"),
    "CI/CD": ("ci cd", "ci/cd", "continuous integration", "continuous deployment"),
    "Java": ("java", "spring boot", "microservices"),
    "API Development": ("api", "rest api", "backend", "microservice"),
    "Testing": ("testing", "selenium", "regression", "test cases", "quality assurance", "qa"),
    "Recruitment": ("recruitment", "talent acquisition", "hiring", "sourcing"),
    "HR Operations": ("onboarding", "payroll", "employee relations", "compliance"),
    "Sales": ("sales", "lead generation", "negotiation", "crm", "account management"),
    "DevOps": ("devops", "infrastructure", "monitoring", "linux", "automation"),
    "Model Deployment": ("model deployment", "deployment", "serving", "mlops"),
    "Analytics": ("analytics", "data analysis", "business intelligence", "feature engineering"),
}
