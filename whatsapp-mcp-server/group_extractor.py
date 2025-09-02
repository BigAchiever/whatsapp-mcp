import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

class GroupMessageExtractor:
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
    
    def extract_group_messages(self, group_jid: str, limit: int = None) -> List[Dict[str, Any]]:
        """Extract all messages from a specific group"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        try:
            # Build query with optional limit
            query = """
                SELECT 
                    id,
                    sender,
                    content,
                    timestamp,
                    is_from_me,
                    media_type,
                    filename
                FROM messages 
                WHERE chat_jid = ?
                ORDER BY timestamp ASC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (group_jid,))
            rows = cursor.fetchall()
            
            messages = []
            for row in rows:
                message = {
                    'id': row[0],
                    'sender': row[1],
                    'content': row[2] if row[2] else '',
                    'timestamp': row[3],
                    'is_from_me': bool(row[4]),
                    'media_type': row[5] if row[5] else None,
                    'filename': row[6] if row[6] else None
                }
                messages.append(message)
            
            print(f"📨 Extracted {len(messages)} messages from group {group_jid}")
            return messages
            
        except Exception as e:
            print(f"❌ Error extracting messages: {e}")
            return []
    
    def clean_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and format messages for better readability"""
        cleaned_messages = []
        
        for msg in messages:
            # Clean content (remove empty strings, normalize)
            content = msg['content'].strip() if msg['content'] else ''
            
            # Format timestamp
            timestamp = msg['timestamp']
            if timestamp:
                try:
                    # Convert to readable format
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    formatted_time = timestamp
            else:
                formatted_time = "Unknown"
            
            # Create clean message
            clean_msg = {
                'id': msg['id'],
                'sender': msg['sender'],
                'content': content,
                'timestamp': formatted_time,
                'is_from_me': msg['is_from_me'],
                'media_type': msg['media_type'],
                'filename': msg['filename'],
                'has_content': bool(content),
                'is_media': bool(msg['media_type'])
            }
            
            cleaned_messages.append(clean_msg)
        
        return cleaned_messages
    
    def export_to_json(self, messages: List[Dict[str, Any]], filename: str):
        """Export messages to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
            print(f"💾 Exported {len(messages)} messages to {filename}")
        except Exception as e:
            print(f"❌ Error exporting to JSON: {e}")
    
    def print_chat_log(self, messages: List[Dict[str, Any]], max_messages: int = 20):
        """Print a readable chat log to console"""
        print(f"\n💬 Chat Log Preview (showing {min(max_messages, len(messages))} messages):")
        print("=" * 80)
        
        for i, msg in enumerate(messages[:max_messages]):
            timestamp = msg['timestamp']
            sender = msg['sender']
            content = msg['content']
            is_me = "👤 YOU" if msg['is_from_me'] else f"👤 {sender}"
            
            if msg['is_media']:
                media_info = f"[{msg['media_type']}] {msg['filename'] or 'No filename'}"
                print(f"{timestamp} | {is_me}: {media_info}")
            elif content:
                print(f"{timestamp} | {is_me}: {content}")
            else:
                print(f"{timestamp} | {is_me}: [Empty message]")
        
        if len(messages) > max_messages:
            print(f"... and {len(messages) - max_messages} more messages")
        print("=" * 80)
    
    def get_all_groups(self) -> List[Dict[str, Any]]:
        """Get all available groups from the database"""
        if not self.conn:
            return []
        
        cursor = self.conn.cursor()
        try:
            # Get all unique group JIDs (ending with @g.us)
            query = """
                SELECT DISTINCT 
                    chat_jid,
                    COUNT(*) as message_count
                FROM messages 
                WHERE chat_jid LIKE '%@g.us'
                GROUP BY chat_jid
                ORDER BY message_count DESC
            """
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            groups = []
            for i, row in enumerate(rows):
                group = {
                    'index': i + 1,
                    'jid': row[0],
                    'message_count': row[1]
                }
                groups.append(group)
            
            return groups
            
        except Exception as e:
            print(f"❌ Error fetching groups: {e}")
            return []
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            print("🔒 Database connection closed")

def main():
    # Database path
    db_path = "../whatsapp-bridge/store/messages.db"
    
    # Create extractor
    extractor = GroupMessageExtractor(db_path)
    
    # Connect to database
    if not extractor.connect():
        return
    
    try:
        # Get all available groups
        print("🔍 Available groups:")
        groups = extractor.get_all_groups()
        
        if not groups:
            print("❌ No groups found in the database")
            return
        
        # Display groups with message counts
        for group in groups:
            print(f"  {group['index']}. {group['jid']} ({group['message_count']} messages)")
        
        # Ask user to select a group
        while True:
            try:
                choice = input(f"\n📋 Select group (1-{len(groups)}) or enter group JID directly: ").strip()
                
                if choice.isdigit():
                    # User selected by number
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(groups):
                        selected_group = groups[choice_num - 1]['jid']
                        break
                    else:
                        print(f"❌ Please enter a number between 1 and {len(groups)}")
                else:
                    # User entered JID directly
                    if choice.endswith('@g.us'):
                        selected_group = choice
                        break
                    else:
                        print("❌ Please enter a valid group JID (ending with @g.us)")
            except ValueError:
                print("❌ Invalid input. Please try again.")
        
        print(f"\n🎯 Selected group: {selected_group}")
        
        # Extract messages
        print(f"🔍 Extracting messages from group: {selected_group}")
        messages = extractor.extract_group_messages(selected_group)
        
        if not messages:
            print("❌ No messages found for this group")
            return
        
        # Clean messages
        print("🧹 Cleaning and formatting messages...")
        cleaned_messages = extractor.clean_messages(messages)
        
        # Show chat log preview
        extractor.print_chat_log(cleaned_messages, max_messages=15)
        
        # Export to JSON
        output_file = f"group_messages_{selected_group.replace('@', '_').replace('.', '_')}.json"
        extractor.export_to_json(cleaned_messages, output_file)
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"  Total messages: {len(messages)}")
        print(f"  Text messages: {sum(1 for m in cleaned_messages if m['has_content'])}")
        print(f"  Media messages: {sum(1 for m in cleaned_messages if m['is_media'])}")
        print(f"  Your messages: {sum(1 for m in cleaned_messages if m['is_from_me'])}")
        print(f"  Exported to: {output_file}")
        
    finally:
        extractor.close()

if __name__ == "__main__":
    main()
