"""
Domain-Specific Rules for PRD Generation and Analysis.
Each domain has specific compliance, security, and functional requirements that agents must consider.
"""

DOMAIN_RULES = {
    "BFSI": {
        "name": "Banking & Financial Services (BFSI)",
        "compliance": [
            "PCI-DSS compliance for payment card data",
            "SOX (Sarbanes-Oxley) for financial reporting",
            "KYC (Know Your Customer) and AML (Anti-Money Laundering) requirements",
            "RBI/FDIC/FCA regulatory guidelines based on region",
            "Basel III capital and liquidity requirements"
        ],
        "security": [
            "Multi-factor authentication (MFA) mandatory for transactions",
            "End-to-end encryption for all financial data",
            "Fraud detection and transaction monitoring",
            "Secure session management with automatic timeout",
            "Audit trails for all financial transactions"
        ],
        "functional": [
            "Real-time transaction processing",
            "Account reconciliation and balance management",
            "Interest calculation and accrual",
            "Statement generation and reporting",
            "Integration with core banking systems"
        ]
    },
    "Healthcare": {
        "name": "Healthcare & Life Sciences",
        "compliance": [
            "HIPAA compliance for Protected Health Information (PHI)",
            "HITECH Act requirements for electronic health records",
            "FDA 21 CFR Part 11 for electronic signatures",
            "HL7/FHIR standards for healthcare interoperability",
            "GDPR for patient data in EU regions"
        ],
        "security": [
            "Role-based access control (RBAC) for patient data",
            "Encryption at rest and in transit for all PHI",
            "Audit logging for all data access",
            "Consent management for data sharing",
            "Breach notification procedures"
        ],
        "functional": [
            "Electronic Health Records (EHR) integration",
            "Clinical decision support systems",
            "Patient scheduling and appointment management",
            "Prescription and medication management",
            "Lab results and diagnostic reporting"
        ]
    },
    "Retail": {
        "name": "Retail & E-Commerce",
        "compliance": [
            "PCI-DSS for payment processing",
            "Consumer protection regulations",
            "GDPR/CCPA for customer data privacy",
            "Accessibility standards (WCAG 2.1)",
            "Return and refund policy compliance"
        ],
        "security": [
            "Secure payment gateway integration",
            "Customer data encryption",
            "Bot protection and fraud prevention",
            "Secure checkout process",
            "Address verification systems"
        ],
        "functional": [
            "Product catalog and inventory management",
            "Shopping cart and checkout flow",
            "Order management and fulfillment",
            "Customer reviews and ratings",
            "Promotions, discounts, and loyalty programs"
        ]
    },
    "Telecom": {
        "name": "Telecommunications",
        "compliance": [
            "TRAI/FCC regulatory compliance",
            "Lawful interception requirements",
            "Number portability regulations",
            "Data retention policies",
            "Net neutrality requirements"
        ],
        "security": [
            "SIM authentication and fraud prevention",
            "Network security and DDoS protection",
            "Customer data privacy protection",
            "Secure API access for partners",
            "Call record encryption"
        ],
        "functional": [
            "Subscriber management and provisioning",
            "Billing and revenue assurance",
            "Network monitoring and QoS",
            "Self-service portal for customers",
            "Usage tracking and quota management"
        ]
    },
    "Manufacturing": {
        "name": "Manufacturing & Supply Chain",
        "compliance": [
            "ISO 9001 quality management standards",
            "ISO 14001 environmental management",
            "OSHA workplace safety requirements",
            "Industry-specific certifications (automotive: IATF 16949)",
            "Export control and trade compliance"
        ],
        "security": [
            "Industrial control system (ICS) security",
            "Supply chain data protection",
            "Intellectual property protection",
            "Vendor access management",
            "Physical and cyber security convergence"
        ],
        "functional": [
            "Production planning and scheduling",
            "Inventory and warehouse management",
            "Quality control and defect tracking",
            "Supplier relationship management",
            "Bill of Materials (BOM) management"
        ]
    },
    "Logistics": {
        "name": "Logistics & Transportation",
        "compliance": [
            "DOT/FMCSA transportation regulations",
            "Customs and import/export compliance",
            "Hazardous materials handling (HAZMAT)",
            "Driver hours of service regulations",
            "Chain of custody documentation"
        ],
        "security": [
            "Real-time tracking data protection",
            "Driver and vehicle authentication",
            "Cargo security and theft prevention",
            "Electronic logging device (ELD) security",
            "API security for partner integrations"
        ],
        "functional": [
            "Route optimization and planning",
            "Real-time shipment tracking",
            "Fleet management and maintenance",
            "Delivery scheduling and proof of delivery",
            "Freight rate calculation and quoting"
        ]
    },
    "Education": {
        "name": "Education & EdTech",
        "compliance": [
            "FERPA for student data privacy",
            "COPPA for children under 13",
            "Section 508 accessibility requirements",
            "State education standards alignment",
            "Accreditation requirements"
        ],
        "security": [
            "Student identity verification",
            "Secure assessment and proctoring",
            "Parental consent management",
            "Data minimization for minors",
            "Safe communication channels"
        ],
        "functional": [
            "Learning Management System (LMS) features",
            "Student enrollment and registration",
            "Grade book and progress tracking",
            "Assignment submission and grading",
            "Virtual classroom and video conferencing"
        ]
    },
    "Insurance": {
        "name": "Insurance",
        "compliance": [
            "State/country insurance regulations",
            "Solvency II requirements (EU)",
            "NAIC model laws and regulations",
            "Anti-fraud regulations",
            "Claims handling guidelines"
        ],
        "security": [
            "Policyholder data encryption",
            "Claims fraud detection",
            "Agent and broker authentication",
            "Document security and e-signatures",
            "Third-party data sharing controls"
        ],
        "functional": [
            "Policy administration and lifecycle",
            "Underwriting and risk assessment",
            "Claims processing and settlement",
            "Premium calculation and billing",
            "Agent/broker portal and commission management"
        ]
    },
    "Government": {
        "name": "Government & Public Sector",
        "compliance": [
            "FedRAMP for cloud services",
            "FISMA security requirements",
            "Section 508 accessibility",
            "Freedom of Information Act (FOIA)",
            "Government procurement regulations"
        ],
        "security": [
            "NIST cybersecurity framework",
            "Multi-level security clearances",
            "Citizen data privacy protection",
            "Secure government network integration",
            "Incident response and reporting"
        ],
        "functional": [
            "Citizen service portals",
            "Case management systems",
            "Document management and workflow",
            "Public records and transparency",
            "Inter-agency data sharing"
        ]
    },
    "IT-SaaS": {
        "name": "Information Technology & SaaS",
        "compliance": [
            "SOC 2 Type II compliance",
            "ISO 27001 information security",
            "GDPR data protection",
            "Service Level Agreement (SLA) requirements",
            "Data residency and sovereignty"
        ],
        "security": [
            "Multi-tenant data isolation",
            "API authentication and rate limiting",
            "SSO and identity federation",
            "Encryption key management",
            "Vulnerability management and patching"
        ],
        "functional": [
            "Subscription and billing management",
            "User provisioning and access control",
            "Usage analytics and reporting",
            "API management and documentation",
            "Multi-tenant architecture support"
        ]
    },
    "AI-Analytics": {
        "name": "Data & Analytics / AI",
        "compliance": [
            "AI Ethics and fairness guidelines",
            "GDPR Article 22 (automated decision-making)",
            "Model explainability requirements",
            "Data lineage and provenance tracking",
            "Bias detection and mitigation"
        ],
        "security": [
            "Training data protection",
            "Model intellectual property security",
            "Secure ML pipeline",
            "Adversarial attack prevention",
            "Data anonymization and pseudonymization"
        ],
        "functional": [
            "Data ingestion and ETL pipelines",
            "Model training and versioning",
            "Real-time inference APIs",
            "Dashboard and visualization",
            "A/B testing and experiment tracking"
        ]
    },
    "Gaming-Media": {
        "name": "Gaming & Media",
        "compliance": [
            "COPPA for games targeting children",
            "Age rating requirements (ESRB, PEGI)",
            "Gambling regulations for in-app purchases",
            "Copyright and content licensing",
            "Broadcasting regulations"
        ],
        "security": [
            "Anti-cheat and fair play systems",
            "DRM and content protection",
            "User account security",
            "In-game purchase fraud prevention",
            "Player data privacy"
        ],
        "functional": [
            "User profile and progression systems",
            "Matchmaking and leaderboards",
            "In-app purchases and virtual currency",
            "Content delivery and streaming",
            "Social features and community management"
        ]
    },
    "Energy": {
        "name": "Energy & Utilities",
        "compliance": [
            "NERC CIP for critical infrastructure",
            "EPA environmental regulations",
            "State utility commission requirements",
            "Smart grid interoperability standards",
            "Renewable energy mandates"
        ],
        "security": [
            "SCADA/ICS system security",
            "Smart meter data protection",
            "Grid cybersecurity",
            "Physical-cyber convergence",
            "Vendor risk management"
        ],
        "functional": [
            "Meter data management",
            "Billing and rate calculation",
            "Outage management and restoration",
            "Demand forecasting and load management",
            "Renewable energy integration"
        ]
    },
    "RealEstate": {
        "name": "Real Estate & Construction",
        "compliance": [
            "Fair Housing Act requirements",
            "RESPA/TILA for mortgage transactions",
            "Building codes and permits",
            "ADA accessibility requirements",
            "Environmental impact regulations"
        ],
        "security": [
            "Transaction escrow security",
            "Document and contract protection",
            "Client financial data security",
            "Property access control systems",
            "Wire fraud prevention"
        ],
        "functional": [
            "Property listing and search",
            "Transaction management and closing",
            "Project management and scheduling",
            "Contractor and vendor management",
            "Inspection and compliance tracking"
        ]
    },
    "Travel": {
        "name": "Travel & Hospitality",
        "compliance": [
            "PCI-DSS for payment processing",
            "GDS/OTA integration standards",
            "Passenger data protection (GDPR, PNR)",
            "Accessibility requirements",
            "Consumer protection for bookings"
        ],
        "security": [
            "Booking fraud prevention",
            "Guest data privacy",
            "Secure payment processing",
            "Identity verification for check-in",
            "Partner API security"
        ],
        "functional": [
            "Booking and reservation management",
            "Inventory and pricing optimization",
            "Guest experience and loyalty programs",
            "Check-in/check-out processes",
            "Reviews and reputation management"
        ]
    },
    "HR-Payroll": {
        "name": "HR & Payroll Systems",
        "compliance": [
            "Labor law compliance (FLSA, state laws)",
            "Equal Employment Opportunity (EEO)",
            "Tax withholding and reporting (W-2, 1099)",
            "Benefits administration (ERISA, ACA)",
            "Employee data privacy laws"
        ],
        "security": [
            "PII and compensation data encryption",
            "Role-based access to employee records",
            "Payroll processing security",
            "Direct deposit verification",
            "Audit logging for all changes"
        ],
        "functional": [
            "Employee onboarding and offboarding",
            "Time and attendance tracking",
            "Payroll processing and tax calculation",
            "Benefits enrollment and management",
            "Performance management and reviews"
        ]
    },
    "FMCG": {
        "name": "FMCG & Consumer Goods",
        "compliance": [
            "FDA/FSSAI food safety regulations",
            "Product labeling requirements",
            "Recall management procedures",
            "Trade promotion compliance",
            "Environmental packaging regulations"
        ],
        "security": [
            "Supply chain traceability",
            "Distributor data protection",
            "Brand protection and anti-counterfeiting",
            "Retailer integration security",
            "Consumer data privacy"
        ],
        "functional": [
            "Distributor and retailer management",
            "Trade promotion planning",
            "Demand forecasting and planning",
            "Secondary sales tracking",
            "Shelf space and merchandising"
        ]
    },
    "CapitalMarkets": {
        "name": "Capital Markets",
        "compliance": [
            "SEC/FINRA regulations",
            "MiFID II requirements (EU)",
            "Best execution obligations",
            "Trade reporting (CFTC, EMIR)",
            "Market manipulation prevention"
        ],
        "security": [
            "Trading system security",
            "Market data protection",
            "Algorithmic trading safeguards",
            "Client asset segregation",
            "Insider trading prevention"
        ],
        "functional": [
            "Order management and execution",
            "Position and risk management",
            "Market data feeds and analytics",
            "Settlement and clearing",
            "Regulatory reporting automation"
        ]
    },
    "Cybersecurity": {
        "name": "Cybersecurity",
        "compliance": [
            "NIST Cybersecurity Framework",
            "ISO 27001/27002 standards",
            "SOC 2 Type II requirements",
            "Industry-specific regulations (HIPAA, PCI)",
            "Breach notification laws"
        ],
        "security": [
            "Zero-trust architecture principles",
            "Threat intelligence integration",
            "Incident response automation",
            "Vulnerability management",
            "Security orchestration (SOAR)"
        ],
        "functional": [
            "Security monitoring and SIEM",
            "Vulnerability scanning and assessment",
            "Identity and access management",
            "Endpoint detection and response",
            "Security awareness training"
        ]
    },
    "Automotive": {
        "name": "Automotive & Mobility",
        "compliance": [
            "UNECE vehicle regulations",
            "ISO 26262 functional safety",
            "Automotive SPICE process standards",
            "Connected car data privacy",
            "Emissions and environmental standards"
        ],
        "security": [
            "Vehicle-to-everything (V2X) security",
            "OTA update integrity",
            "In-vehicle network security (CAN bus)",
            "Telematics data protection",
            "Supply chain cybersecurity"
        ],
        "functional": [
            "Vehicle connectivity and telematics",
            "Fleet management systems",
            "Dealer management systems",
            "Service and maintenance scheduling",
            "EV charging network integration"
        ]
    }
}


def get_domain_guidance(domain: str) -> str:
    """
    Returns a formatted string of domain-specific guidance for agent prompts.
    """
    rules = DOMAIN_RULES.get(domain, None)
    if not rules:
        return f"Domain: {domain} (general guidance applies)"
    
    guidance = f"""
DOMAIN: {rules['name']}

MANDATORY COMPLIANCE REQUIREMENTS:
{chr(10).join(f'- {item}' for item in rules['compliance'])}

SECURITY REQUIREMENTS:
{chr(10).join(f'- {item}' for item in rules['security'])}

EXPECTED FUNCTIONAL AREAS:
{chr(10).join(f'- {item}' for item in rules['functional'])}
"""
    return guidance.strip()


def get_all_domains() -> list[str]:
    """Returns list of all supported domain keys."""
    return list(DOMAIN_RULES.keys())
