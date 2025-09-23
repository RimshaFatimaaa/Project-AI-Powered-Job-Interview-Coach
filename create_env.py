"""
Script to create .env file with proper UTF-8 encoding
"""
import os

# Environment variables content
env_content = """SUPABASE_URL=https://sbwjxrvqcbwojkmlmgdu.supabase.co
SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNid2p4cnZxY2J3b2prbWxtZ2R1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTgyMDU1NDUsImV4cCI6MjA3Mzc4MTU0NX0.LKGGysulqMV2TGfJuCxLtnQ_KwW6voamQNhvKq0WhKo
OPENAI_API_KEY=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
"""

# Write to .env file with UTF-8 encoding
with open('.env', 'w', encoding='utf-8') as f:
    f.write(env_content)

print("✅ .env file created successfully with UTF-8 encoding")
