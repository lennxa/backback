# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import re
app = FastAPI()
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from datetime import datetime, timedelta
import random
from typing import List, Dict, Any, Optional

# Admin dashboard models
class ClaimMetrics(BaseModel):
    total_claims: int
    pending_review: int
    flagged_claims: int
    active_users: int

class MonthlyClaimData(BaseModel):
    month: str
    count: int

class FraudMetrics(BaseModel):
    fraud_detected: int
    suspicious_claims: int
    false_positives: int
    accuracy: float

class RecentClaim(BaseModel):
    claim_id: str
    claimant: str
    amount: float
    date: str
    status: str
    flag: Optional[str] = None

class DashboardData(BaseModel):
    metrics: ClaimMetrics
    monthly_claims: List[MonthlyClaimData]
    fraud_metrics: FraudMetrics
    recent_claims: List[RecentClaim]

@app.get("/api/admin/dashboard", response_model=DashboardData)
async def get_dashboard_data():
    """Endpoint to retrieve dashboard data for admin view"""
    
    # Generate mock data for dashboard metrics
    total_claims = random.randint(950, 1200)
    pending_review = random.randint(80, 150)
    flagged_claims = random.randint(20, 50)
    active_users = random.randint(500, 800)
    
    # Generate mock monthly claims data (last 6 months)
    current_date = datetime.now()
    monthly_claims = []
    
    for i in range(5, -1, -1):
        month_date = current_date - timedelta(days=30 * i)
        month_name = month_date.strftime("%b")
        monthly_claims.append(
            MonthlyClaimData(
                month=month_name,
                count=random.randint(150, 300)
            )
        )
    
    # Generate mock fraud metrics
    fraud_detected = random.randint(15, 40)
    suspicious_claims = random.randint(40, 80)
    false_positives = random.randint(5, 15)
    accuracy = round(random.uniform(0.92, 0.98), 2)
    
    # Generate mock recent claims
    statuses = ["Approved", "Pending", "Under Review", "Denied"]
    flags = [None, "Suspicious Amount", "Duplicate Claim", "Incomplete Info", "Expired Policy"]
    
    recent_claims = []
    for i in range(10):
        status = random.choice(statuses)
        flag = random.choice(flags) if status in ["Pending", "Under Review"] and random.random() > 0.5 else None
        
        claim_date = current_date - timedelta(days=random.randint(0, 14))
        
        recent_claims.append(
            RecentClaim(
                claim_id=f"CLAIM-{random.randint(10000, 99999)}",
                claimant=random.choice([
                    "John Smith", "Emma Johnson", "Michael Brown", "Sophia Williams", 
                    "James Miller", "Olivia Davis", "Robert Wilson", "Emily Taylor",
                    "David Anderson", "Ava Martinez", "Christopher Lee", "Mia Garcia"
                ]),
                amount=round(random.uniform(100, 5000), 2),
                date=claim_date.strftime("%Y-%m-%d"),
                status=status,
                flag=flag
            )
        )
    
    # Sort recent claims by date (newest first)
    recent_claims.sort(key=lambda x: x.date, reverse=True)
    
    return DashboardData(
        metrics=ClaimMetrics(
            total_claims=total_claims,
            pending_review=pending_review,
            flagged_claims=flagged_claims,
            active_users=active_users
        ),
        monthly_claims=monthly_claims,
        fraud_metrics=FraudMetrics(
            fraud_detected=fraud_detected,
            suspicious_claims=suspicious_claims,
            false_positives=false_positives,
            accuracy=accuracy
        ),
        recent_claims=recent_claims
    )


from datetime import datetime
import json
import os
from pathlib import Path
# Create a model for the form data
class ClaimFormData(BaseModel):
    fullname: str
    dob: str
    phone: str
    email: str
    policy: str
    group: Optional[str] = None
    provider: str
    dos: str
    providerName: str
    diagnosis: str
    procedure: str
    amount: str
    comments: Optional[str] = None
    terms: bool
class ClaimResponse(BaseModel):
    success: bool
    message: str
    claim_id: Optional[str] = None
# Create a directory for storing claims if it doesn't exist
claims_dir = Path("claims")
claims_dir.mkdir(exist_ok=True)
@app.post("/api/submit-claim", response_model=ClaimResponse)
async def submit_claim(claim_data: ClaimFormData):
    """
    Endpoint to receive and store claim form submissions
    """
    try:
        # Generate a unique claim ID based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        claim_id = f"CLAIM-{timestamp}"
        # Add the claim ID to the data
        claim_data_dict = claim_data.dict()
        claim_data_dict["claim_id"] = claim_id
        claim_data_dict["submission_date"] = datetime.now().isoformat()
        # Store the claim in a JSON file
        file_path = claims_dir / f"{claim_id}.json"
        with open(file_path, "w") as f:
            json.dump(claim_data_dict, f, indent=2)
        # In a real application, you might also:
        # - Store in a database
        # - Send email notifications
        # - Trigger workflow processes
        return ClaimResponse(
            success=True,
            message="Claim submitted successfully",
            claim_id=claim_id
        )
    except Exception as e:
        # Log the error for server-side debugging
        print(f"Error processing claim submission: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your claim"
        )
import os
import requests
from dotenv import load_dotenv
# Add these new models
class Message(BaseModel):
    role: str
    text: str
class GeminiRequest(BaseModel):
    conversation_history: List[Message]
class GeminiResponse(BaseModel):
    response_text: str
# Load environment variables (you'll store your API key here)
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Add this new endpoint
@app.post("/api/gemini", response_model=GeminiResponse)
async def call_gemini(request: GeminiRequest):
    """Proxy endpoint to call Gemini API"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    # Format the conversation history for Gemini API
    system_instruction = request.conversation_history[0].text if request.conversation_history else ""
    request_payload = {
        "system_instruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [
            {
                "role": msg.role,
                "parts": [{"text": msg.text}]
            }
            for msg in request.conversation_history[1:]
        ]
    }
    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}",
            json=request_payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        data = response.json()
        if "candidates" in data and len(data["candidates"]) > 0:
            response_text = data["candidates"][0]["content"]["parts"][0]["text"]
            return GeminiResponse(response_text=response_text)
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format from Gemini API")
    except requests.RequestException as e:
        print("something went wrong in gemini")
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")
# Sample MetLife policy document broken into chunks with IDs
policy_chunks = [
    {
        "id": "section_1",
        "title": "Policy Overview",
        "text": "This MetLife insurance policy (Policy #ML-2023-45678) provides comprehensive health coverage for the policyholder and eligible dependents. The effective date of this policy is January 1, 2023, and it remains in force until December 31, 2023, unless terminated earlier under the provisions described herein."
    },
    {
        "id": "section_2",
        "title": "Eligibility Requirements",
        "text": "To be eligible for coverage under this policy, the primary insured must be a full-time employee working at least 30 hours per week. Dependents eligible for coverage include the legal spouse of the primary insured and unmarried children under the age of 26."
    },
    {
        "id": "section_3",
        "title": "Premium Payments",
        "text": "Premium payments are due on the first day of each month. A grace period of 30 days is provided, during which coverage will continue uninterrupted. Failure to pay premiums by the end of the grace period will result in termination of coverage."
    },
    {
        "id": "section_4",
        "title": "Covered Services",
        "text": "This policy covers preventive care services at 100% when rendered by in-network providers. These services include annual physicals, immunizations, and routine screenings as recommended by the U.S. Preventive Services Task Force."
    },
    {
        "id": "section_5",
        "title": "Hospital Services",
        "text": "Inpatient hospital services are covered at 80% after the deductible has been met. Pre-authorization is required for all non-emergency hospital admissions. Failure to obtain pre-authorization may result in a 50% reduction of benefits."
    },
    {
        "id": "section_6",
        "title": "Prescription Drug Benefits",
        "text": "Prescription medications are categorized into three tiers: generic ($10 copay), preferred brand ($30 copay), and non-preferred brand ($50 copay). Specialty medications require prior authorization and are subject to 20% coinsurance up to a maximum of $250 per prescription."
    },
    {
        "id": "section_7",
        "title": "Mental Health Services",
        "text": "Mental health and substance abuse services are covered on parity with medical services. Outpatient therapy visits require a $25 copay per session. Inpatient mental health treatment is subject to the same coverage rules as other inpatient services."
    },
    {
        "id": "section_8",
        "title": "Emergency Services",
        "text": "Emergency room visits are covered with a $150 copay, which is waived if the patient is admitted to the hospital. Ambulance services are covered at 80% after the deductible for medically necessary transportation."
    },
    {
        "id": "section_9",
        "title": "Out-of-Network Coverage",
        "text": "Services provided by out-of-network providers are generally covered at 60% of the allowed amount after a separate out-of-network deductible has been met. The policyholder is responsible for the remaining 40% plus any charges exceeding the allowed amount."
    },
    {
        "id": "section_10",
        "title": "Exclusions",
        "text": "This policy does not cover cosmetic procedures, experimental treatments, or services deemed not medically necessary. Long-term care, custodial care, and non-prescription medications are also excluded from coverage."
    },
    {
        "id": "section_11",
        "title": "Claims Submission",
        "text": "All claims must be submitted within 90 days of the date of service. In-network providers will submit claims directly to MetLife. For out-of-network services, the policyholder is responsible for submitting itemized bills and completed claim forms."
    },
    {
        "id": "section_12",
        "title": "Appeals Process",
        "text": "If a claim is denied, the policyholder has the right to appeal the decision within 180 days. The appeal must be submitted in writing and include the policy number, date of service, and reason for the appeal."
    },
    {
        "id": "section_13",
        "title": "Termination of Coverage",
        "text": "This policy may be terminated for non-payment of premiums, fraud, or material misrepresentation. Coverage will also terminate when the policyholder is no longer eligible under the terms of the policy. Dependents will lose coverage when they no longer meet eligibility requirements."
    },
    {
        "id": "section_14",
        "title": "Continuation of Coverage",
        "text": "Under certain circumstances, policyholders may be eligible to continue coverage through COBRA for up to 18 months after employment termination. Notification of COBRA eligibility will be provided within 14 days of the qualifying event."
    },
    {
        "id": "section_15",
        "title": "Policy Amendments",
        "text": "MetLife reserves the right to amend the terms of this policy with 60 days written notice to the policyholder. Any changes to benefits or coverages will be clearly outlined in the amendment notice."
    }
]
class Query(BaseModel):
    query: str
class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    context: str

# Add this model for policy data
class PolicySummary(BaseModel):
    policy_id: str
    policy_name: str
    policy_number: str
    policy_type: str
    start_date: str
    end_date: str
    status: str

@app.get("/api/policies", response_model=List[PolicySummary])
async def get_policies():
    """Endpoint to retrieve all policies for the current user"""
    # In a real application, you would fetch this data from a database
    # and filter based on the authenticated user

    # For now, return the same policy multiple times with different IDs
    sample_policies = [
        PolicySummary(
            policy_id=f"policy_{i}",
            policy_name="MetLife Health Insurance",
            policy_number=f"ML-2023-{45678+i}",
            policy_type="Health",
            start_date="2023-01-01",
            end_date="2023-12-31",
            status="Active" if i % 5 != 0 else "Pending Renewal"
        )
        for i in range(5)
    ]

    # Add a few different policy types
    sample_policies.append(
        PolicySummary(
            policy_id="policy_dental",
            policy_name="MetLife Dental Plan",
            policy_number="ML-DENT-78945",
            policy_type="Dental",
            start_date="2023-01-01",
            end_date="2023-12-31",
            status="Active"
        )
    )

    sample_policies.append(
        PolicySummary(
            policy_id="policy_vision",
            policy_name="MetLife Vision Care",
            policy_number="ML-VIS-12345",
            policy_type="Vision",
            start_date="2023-01-01",
            end_date="2023-12-31",
            status="Active"
        )
    )

    return sample_policies

# Add this endpoint to get details for a specific policy
@app.get("/api/policies/{policy_id}")
async def get_policy_details(policy_id: str):
    """Get detailed information for a specific policy"""
    # In a real application, you would fetch this data from a database

    # For this example, we'll just return the same policy document for all IDs
    return {
        "policy_id": policy_id,
        "policy_name": "MetLife Health Insurance",
        "policy_number": f"ML-2023-{policy_id.split('_')[-1]}" if policy_id.startswith("policy_") else "ML-2023-45678",
        "policy_type": "Health",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "status": "Active",
        "sections": policy_chunks  # Reuse the existing policy chunks
    }

# Initialize the sentence transformer model
# Load the model just once when the server starts
model = SentenceTransformer('all-MiniLM-L6-v2')
# Pre-compute embeddings for all policy chunks
policy_texts = [f"{chunk['title']}: {chunk['text']}" for chunk in policy_chunks]
policy_embeddings = model.encode(policy_texts, convert_to_tensor=True)
# Move to CPU if needed
if torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    policy_embeddings_np = [embedding.detach().cpu().numpy() for embedding in policy_embeddings]
else:
    policy_embeddings_np = [embedding.numpy() for embedding in policy_embeddings]
@app.on_event("startup")
async def startup_event():
    print("Sentence Transformer model loaded successfully.")
    # Log device info for debugging
    if torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        print("Using CPU for tensor operations")
def get_relevant_chunks_with_embeddings(query: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Use sentence embeddings to find the most relevant policy chunks
    """
    # Encode the query
    query_embedding = model.encode(query, convert_to_tensor=True)
    # Move query embedding to CPU and convert to numpy
    if torch.cuda.is_available() or hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        query_embedding_np = query_embedding.detach().cpu().numpy()
    else:
        query_embedding_np = query_embedding.numpy()
    # Calculate cosine similarity between query and all chunks
    similarities = []
    for i, embedding_np in enumerate(policy_embeddings_np):
        # Calculate cosine similarity using numpy
        similarity = np.dot(query_embedding_np, embedding_np) / (
            np.linalg.norm(query_embedding_np) * np.linalg.norm(embedding_np)
        )
        similarities.append((similarity, i))
    # Sort by similarity score (highest first)
    similarities.sort(reverse=True)
    # Return the top N most relevant chunks
    relevant_chunks = []
    for _, chunk_index in similarities[:top_n]:
        relevant_chunks.append(policy_chunks[chunk_index])
    return relevant_chunks
def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into context string with source IDs for the LLM"""
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"<source_id>{chunk['id']}</source_id>\n{chunk['title']}: {chunk['text']}")
    return "\n\n".join(context_parts)
@app.post("/api/policy-qa", response_model=ChunkResponse)
async def policy_qa(query: Query):
    if not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    relevant_chunks = get_relevant_chunks_with_embeddings(query.query)
    if not relevant_chunks:
        return ChunkResponse(chunks=[], context="No relevant information found in the policy document.")
    context = format_context(relevant_chunks)
    return ChunkResponse(chunks=relevant_chunks, context=context)
# Run with: uvicorn main:app --reload --port 8000
