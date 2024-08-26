from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import env

if env.mode == "dsv":
    DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(DATABASE_URL)
else:
    DATABASE_URL = f"mysql+pymysql://{env.db_username}:{env.db_password}@{env.db_address}:{env.db_port}/{env.db_name}"
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,         
        max_overflow=20,      
        pool_timeout=30,      
        pool_recycle=1800,    
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
