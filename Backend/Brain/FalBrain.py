import os
import datetime
import sqlite3
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ConversationEntry:
    """Data class for conversation entries"""
    id: Optional[int] = None
    user_message: str = ""
    assistant_message: Optional[str] = None
    timestamp: Optional[datetime.datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class DatabaseManager:
    """Advanced database manager with connection pooling and optimized queries"""
    
    def __init__(self, db_path: str = 'Database/History/Falcon.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_database(self):
        """Initialize database with optimized schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table with better indexing
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT DEFAULT '{}',
                    message_length INTEGER,
                    response_time REAL
                )
            ''')
            
            # Tags table for categorization
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    tag_name TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            ''')
            
            # Analytics table for performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    total_conversations INTEGER DEFAULT 0,
                    avg_response_time REAL DEFAULT 0.0,
                    error_count INTEGER DEFAULT 0
                )
            ''')
            
            # Create indices for better performance
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
                ON conversations(timestamp)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversations_search 
                ON conversations(user_message, assistant_message)
            ''')
            
            conn.commit()
    
    def add_conversation(self, entry: ConversationEntry) -> int:
        """Add a new conversation entry"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations (user_message, assistant_message, metadata, message_length)
                VALUES (?, ?, ?, ?)
            ''', (
                entry.user_message,
                entry.assistant_message,
                json.dumps(entry.metadata or {}),
                len(entry.user_message)
            ))
            conversation_id = cursor.lastrowid
            conn.commit()
            return conversation_id
    
    def update_assistant_response(self, conversation_id: int, response: str, response_time: float = 0.0):
        """Update assistant response with performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE conversations 
                SET assistant_message = ?, response_time = ?
                WHERE id = ?
            ''', (response, response_time, conversation_id))
            conn.commit()
    
    def get_conversation_history(self, limit: Optional[int] = None, include_metadata: bool = False) -> List[Dict[str, Any]]:
        """Get conversation history with optional metadata"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT user_message, assistant_message, timestamp, metadata
                FROM conversations 
                WHERE assistant_message IS NOT NULL
                ORDER BY timestamp ASC
            '''
            
            if limit:
                query += ' LIMIT ?'
                cursor.execute(query, (limit,))
            else:
                cursor.execute(query)
            
            messages = []
            for row in cursor.fetchall():
                messages.append({"role": "user", "content": row['user_message']})
                if row['assistant_message']:
                    messages.append({"role": "assistant", "content": row['assistant_message']})
                    
                if include_metadata and row['metadata']:
                    try:
                        metadata = json.loads(row['metadata'])
                        messages[-1]['metadata'] = metadata
                    except json.JSONDecodeError:
                        pass
            
            return messages
    
    def search_conversations(self, keyword: str, limit: int = 50) -> List[Tuple]:
        """Advanced search with ranking"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_message, assistant_message, timestamp,
                       (CASE 
                        WHEN user_message LIKE ? THEN 2
                        WHEN assistant_message LIKE ? THEN 1
                        ELSE 0
                       END) as relevance
                FROM conversations 
                WHERE user_message LIKE ? OR assistant_message LIKE ?
                ORDER BY relevance DESC, timestamp DESC
                LIMIT ?
            ''', (f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', f'%{keyword}%', limit))
            
            return cursor.fetchall()
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total conversations
            cursor.execute('SELECT COUNT(*) FROM conversations')
            total_conversations = cursor.fetchone()[0]
            
            # Average response time
            cursor.execute('SELECT AVG(response_time) FROM conversations WHERE response_time > 0')
            avg_response_time = cursor.fetchone()[0] or 0.0
            
            # Recent activity (last 7 days)
            cursor.execute('''
                SELECT COUNT(*) FROM conversations 
                WHERE timestamp > datetime('now', '-7 days')
            ''')
            recent_activity = cursor.fetchone()[0]
            
            return {
                'total_conversations': total_conversations,
                'avg_response_time': round(avg_response_time, 2),
                'recent_activity': recent_activity
            }

class ContextManager:
    """Manages conversation context and memory"""
    
    def __init__(self, max_context_length: int = 8000):
        self.max_context_length = max_context_length
        self.context_buffer = []
    
    def add_to_context(self, message: Dict[str, Any]):
        """Add message to context with smart truncation"""
        self.context_buffer.append(message)
        
        # Smart truncation based on total length
        total_length = sum(len(msg.get('content', '')) for msg in self.context_buffer)
        while total_length > self.max_context_length and len(self.context_buffer) > 2:
            removed = self.context_buffer.pop(0)
            total_length -= len(removed.get('content', ''))
    
    def get_context(self) -> List[Dict[str, Any]]:
        """Get current context"""
        return self.context_buffer.copy()
    
    def clear_context(self):
        """Clear context buffer"""
        self.context_buffer.clear()

class FalconAI:
    """Advanced AI Assistant with improved capabilities"""
    
    def __init__(self, model: str = "llama3-70b-8192"):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.db = DatabaseManager()
        self.context_manager = ContextManager()
        
        self.system_prompt = {
            "role": "system", 
            "content": """
You are Falcon, I am {Your Name}, an advanced AI assistant created by Utkarsh Rishi to be helpful, creative, and reliable.

Core Guidelines:
- Provide concise, accurate responses (1-2 sentences when possible)
- Always prioritize user safety and provide well-structured information
- Reply in English, even if the question is in Hindi
- Be professional yet engaging
- Learn from conversations to improve over time
- Never mention training data limitations

Location: India 🇮🇳
Creator: Utkarsh Rishi

Remember: Quality over quantity in responses!
            """
        }
    
    def get_real_time_info(self) -> Dict[str, str]:
        """Get comprehensive real-time information"""
        now = datetime.datetime.now()
        return {
            "day": now.strftime("%A"),
            "date": now.strftime("%d"),
            "month": now.strftime("%B"),
            "year": now.strftime("%Y"),
            "time": now.strftime("%H:%M:%S"),
            "timezone": "IST (UTC+5:30)",
            "timestamp": now.isoformat()
        }
    
    def _prepare_messages(self, user_input: str) -> List[Dict[str, Any]]:
        """Prepare messages for API call"""
        messages = [self.system_prompt]
        
        # Add real-time context
        time_info = self.get_real_time_info()
        time_context = {
            "role": "system", 
            "content": f"Current time info: {time_info}"
        }
        messages.append(time_context)
        
        # Add conversation history (last 10 exchanges)
        history = self.db.get_conversation_history(limit=20)
        messages.extend(history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    async def process_message_async(self, user_input: str) -> str:
        """Async message processing"""
        return await asyncio.to_thread(self.process_message, user_input)
    
    def process_message(self, user_input: str) -> str:
        """Process user message with advanced error handling"""
        if not user_input.strip():
            return "Please provide a message! 💬"
        
        start_time = datetime.datetime.now()
        
        try:
            # Create conversation entry
            entry = ConversationEntry(user_message=user_input)
            conversation_id = self.db.add_conversation(entry)
            
            # Prepare messages
            messages = self._prepare_messages(user_input)
            
            # Make API call with streaming
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                stream=True,
                stop=None
            )
            
            # Collect streamed response
            response_parts = []
            for chunk in completion:
                if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    response_parts.append(chunk.choices[0].delta.content)
            
            response = ''.join(response_parts).strip()
            
            if not response:
                response = "I'm having trouble generating a response right now. Please try again! 🔄"
            
            # Calculate response time
            response_time = (datetime.datetime.now() - start_time).total_seconds()
            
            # Update database
            self.db.update_assistant_response(conversation_id, response, response_time)
            
            # Update context
            self.context_manager.add_to_context({"role": "user", "content": user_input})
            self.context_manager.add_to_context({"role": "assistant", "content": response})
            
            return response
            
        except Exception as e:
            error_response = f"Sorry, I encountered an error: {str(e)[:100]}... 🚨"
            try:
                self.db.update_assistant_response(conversation_id, error_response)
            except:
                pass
            return error_response
    
    def search_conversations(self, keyword: str) -> List[Tuple]:
        """Search through conversation history"""
        return self.db.search_conversations(keyword)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get conversation analytics"""
        return self.db.get_analytics()
    
    def clear_context(self):
        """Clear conversation context"""
        self.context_manager.clear_context()

# Main interface functions
def create_assistant() -> FalconAI:
    """Create and return a new FalconAI instance"""
    return FalconAI()

def chat_with_falcon(prompt: str, assistant: Optional[FalconAI] = None) -> str:
    """Chat with Falcon AI assistant"""
    if assistant is None:
        assistant = create_assistant()
    
    return assistant.process_message(prompt)

async def chat_with_falcon_async(prompt: str, assistant: Optional[FalconAI] = None) -> str:
    """Async chat with Falcon AI assistant"""
    if assistant is None:
        assistant = create_assistant()
    
    return await assistant.process_message_async(prompt)

def main_chat_with_falcon(text):
    assistant = create_assistant()

    user_input = text
    
    print(chat_with_falcon(user_input, assistant))