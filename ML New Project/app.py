from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from apscheduler.schedulers.background import BackgroundScheduler
from passlib.context import CryptContext
import joblib
import numpy as np
import datetime

# ==========================================
# 1. APP & SECURITY INITIALIZATION
# ==========================================
app = FastAPI(title="Nexus Health API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ==========================================
# 2. DATABASE SETUP (SQLite)
# ==========================================
SQLALCHEMY_DATABASE_URL = "sqlite:///./nexus_health.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    # Storing latest metrics for reminders
    latest_bmi_class = Column(Integer, default=2)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 3. BACKGROUND SCHEDULER (REMINDERS)
# ==========================================
def daily_health_routine():
    print(f"[{datetime.datetime.now()}] NEXUS SYSTEM: Running daily AI health scans and sending user reminders...")
    # In a production app, this would loop through the database and send emails/push notifications.

scheduler = BackgroundScheduler()
scheduler.add_job(daily_health_routine, 'interval', hours=24) # Runs every 24 hours
scheduler.start()

# ==========================================
# 4. MACHINE LEARNING MODEL SETUP
# ==========================================
try:
    model = joblib.load('bmi_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("WARNING: ML Model files not found. Please run train_model.py first.")

bmi_categories = {
    0: 'Extremely Weak', 1: 'Weak', 2: 'Normal', 
    3: 'Overweight', 4: 'Obesity', 5: 'Extremely Obese'
}

# ==========================================
# 5. DATA MODELS (PYDANTIC)
# ==========================================
class UserRegister(BaseModel):
    name: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class MLInput(BaseModel):
    height: float
    weight: float
    age: int
    gender: str 

# ==========================================
# 6. AUTHENTICATION ROUTES (Login/Register)
# ==========================================
@app.post("/register")
def register_user(user: UserRegister, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_pw = pwd_context.hash(user.password)
    new_user = UserDB(name=user.name, email=user.email, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully", "user": user.name}

@app.post("/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(UserDB).filter(UserDB.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # In a real app, you would return a JWT Token here. For now, we return success.
    return {"message": "Login successful", "user": db_user.name}

# ==========================================
# 7. AI INFERENCE ROUTE (Dashboard Logic)
# ==========================================
@app.post("/predict")
def predict_bmi(data: MLInput):
    # Scale Data
    input_data = np.array([[data.height, data.weight]])
    scaled_input = scaler.transform(input_data)
    
    # ML Inference
    prediction = int(model.predict(scaled_input)[0])
    probabilities = model.predict_proba(scaled_input)[0]
    confidence = max(probabilities) * 100 
    
    importances = model.feature_importances_
    height_impact = importances[0] * 100
    weight_impact = importances[1] * 100
    
    # Calculate BMR
    if data.gender.lower() == 'male':
        bmr = (10 * data.weight) + (6.25 * data.height) - (5 * data.age) + 5
    else:
        bmr = (10 * data.weight) + (6.25 * data.height) - (5 * data.age) - 161
        
    # Generate AI Action Plan
    tdee = bmr * 1.2 
    if prediction < 2: 
        goal = "Build Mass & Muscle"
        target_cals = int(tdee + 400) 
        p_pct, c_pct, f_pct = 30, 50, 20
        exercise = "Moderate (120 mins/week). Focus on Heavy Weightlifting to convert calories to muscle."
    elif prediction == 2: 
        goal = "Maintain Optimal Health"
        target_cals = int(tdee) 
        p_pct, c_pct, f_pct = 30, 40, 30
        exercise = "Active (150 mins/week). Mix of Light Cardio and Bodyweight Exercises."
    else: 
        goal = "Reduce BMI & Burn Fat"
        target_cals = int(tdee - 500) 
        p_pct, c_pct, f_pct = 40, 30, 30 
        exercise = "Intense (250 mins/week). Focus on HIIT (High Intensity Interval Training) and Brisk Walking."

    protein_g = int((target_cals * (p_pct/100)) / 4)
    carbs_g = int((target_cals * (c_pct/100)) / 4)
    fat_g = int((target_cals * (f_pct/100)) / 9)

    return {
        "index_value": prediction,
        "category": bmi_categories[prediction],
        "confidence": round(confidence, 1),
        "bmr": round(bmr),
        "impact": {"height": round(height_impact, 1), "weight": round(weight_impact, 1)},
        "action_plan": {
            "goal": goal,
            "target_calories": target_cals,
            "macros": {"protein": protein_g, "carbs": carbs_g, "fat": fat_g},
            "percentages": {"p": p_pct, "c": c_pct, "f": f_pct},
            "exercise": exercise
        }
    }