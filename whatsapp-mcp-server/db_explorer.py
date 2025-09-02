import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class WhatsAppDBExplorer:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def get_table_info(self):
        """Get information about all tables in the database"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("\n📊 Database Tables:")
        for table in tables:
            table_name = table[0]
            print(f"  - {table_name}")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            print(f"    Columns:")
            for col in columns:
                print(f"      {col[1]} ({col[2]})")
            print()
        
        return [table[0] for table in tables]
    
    def get_message_count(self, table_name: str = "messages") -> int:
        """Get total message count from a table"""
        if not self.conn:
            return 0
        
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            return count
        except Exception as e:
            print(f"❌ Error counting messages: {e}")
            return 0
    
    def get_sample_messages(self, table_name: str = "messages", limit: int = 5) -> List[Dict[str, Any]]:
        """Get sample messages to understand the structure"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                message = dict(zip(columns, row))
                messages.append(message)
            
            return messages
        except Exception as e:
            print(f"❌ Error fetching sample messages: {e}")
            return []
    
    def get_group_messages(self, group_jid: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get messages from a specific group"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        try:
            # Try to find the right column for group JID
            cursor.execute("PRAGMA table_info(messages);")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # Look for common column names that might contain group JID
            possible_columns = ['chat_jid', 'group_jid', 'jid', 'chat_id', 'group_id']
            chat_column = None
            
            for col in possible_columns:
                if col in column_names:
                    chat_column = col
                    break
            
            if not chat_column:
                print("❌ Could not find group JID column. Available columns:")
                for col in column_names:
                    print(f"  - {col}")
                return []
            
            query = f"SELECT * FROM messages WHERE {chat_column} = ? LIMIT {limit}"
            cursor.execute(query, (group_jid,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                message = dict(zip(columns, row))
                messages.append(message)
            
            return messages
            
        except Exception as e:
            print(f"❌ Error fetching group messages: {e}")
            return []
    
    def analyze_message_structure(self, messages: List[Dict[str, Any]]):
        """Analyze the structure of messages to understand important fields"""
        if not messages:
            print("❌ No messages to analyze")
            return
        
        print(f"\n🔍 Analyzing {len(messages)} messages:")
        
        # Get all unique keys
        all_keys = set()
        for message in messages:
            all_keys.update(message.keys())
        
        print(f"📋 All available fields: {sorted(all_keys)}")
        
        # Analyze each field
        for key in sorted(all_keys):
            print(f"\n📝 Field: {key}")
            
            # Get sample values
            sample_values = []
            for message in messages:
                if key in message and message[key] is not None:
                    value = str(message[key])
                    if len(value) > 50:
                        value = value[:50] + "..."
                    sample_values.append(value)
            
            if sample_values:
                print(f"  Sample values: {sample_values[:3]}")
                print(f"  Non-null count: {len(sample_values)}/{len(messages)}")
            else:
                print(f"  All values are null")
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("🔒 Database connection closed")

def main():
    # Database path
    db_path = "../whatsapp-bridge/store/messages.db"
    
    # Create explorer
    explorer = WhatsAppDBExplorer(db_path)
    
    # Connect to database
    if not explorer.connect():
        return
    
    try:
        # Get table information
        tables = explorer.get_table_info()
        
        # Get message count
        message_count = explorer.get_message_count()
        print(f"📊 Total messages in database: {message_count}")
        
        # Get sample messages
        sample_messages = explorer.get_sample_messages()
        if sample_messages:
            print(f"\n📨 Sample messages structure:")
            explorer.analyze_message_structure(sample_messages)
        
        # Try to get group messages (you'll need to provide a group JID)
        # Example: explorer.get_group_messages("120363023766549700@g.us")
        
    finally:
        explorer.close()

if __name__ == "__main__":
    main()
