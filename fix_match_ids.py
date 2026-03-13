import os
import sys

# Add the backend directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))

from database import get_connection

def fix_match_ids():
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Get all predictions missing a match_id
        cur.execute("SELECT id, home_team, away_team, league, match_date FROM prediction_log WHERE match_id IS NULL")
        missing = cur.fetchall()
        
        updated = 0
        for row in missing:
            # Find matching match_id
            cur.execute("""
                SELECT m.id 
                FROM matches m
                JOIN teams ht ON ht.id = m.home_team_id
                JOIN teams at ON at.id = m.away_team_id
                WHERE ht.name = %s AND at.name = %s
            """, (row['home_team'], row['away_team']))
            
            match = cur.fetchone()
            if match:
                cur.execute("UPDATE prediction_log SET match_id = %s WHERE id = %s", (match['id'], row['id']))
                updated += 1
                
        conn.commit()
        print(f"✅ Successfully backfilled {updated} missing match_ids in prediction_log.")
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == '__main__':
    fix_match_ids()
