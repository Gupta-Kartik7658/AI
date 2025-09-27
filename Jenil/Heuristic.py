import heapq
from typing import List, Tuple, Optional
import re
import os


def preprocess_document(text):
    """
    Preprocess a document:
    1. Normalize (lowercase + remove punctuation except periods)
    2. Tokenize into sentences
    3. Join sentences back into a single string with periods
    """
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s.]', '', text)  # Remove punctuation except periods
    # Replace line breaks with space
    text = text.replace('\n', ' ')
    # Split into sentences by period
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    # Join sentences back into a single string separated by periods
    result = '. '.join(sentences) + '.'
    return result

class Node:
    """
    Node class for A* search algorithm
    Represents a state in the text alignment problem
    """
    def __init__(self, state: Tuple[int, int], parent: Optional['Node'] = None, 
                 g_cost: float = 0, h_cost: float = 0):
        """
        Initialize a node
        
        Args:
            state: (i, j) where i is position in doc1, j is position in doc2
            parent: Parent node in the search tree
            g_cost: Cost from start to this node
            h_cost: Heuristic cost from this node to goal
        """
        self.state = state  # (i, j) positions in both documents
        self.parent = parent
        self.g_cost = g_cost  # Actual cost from start
        self.h_cost = h_cost  # Heuristic cost to goal
        self.f_cost = g_cost + h_cost  # Total cost f(n) = g(n) + h(n)
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        """For node equality comparison"""
        return isinstance(other, Node) and self.state == other.state
    
    def __hash__(self):
        """For using nodes in sets"""
        return hash(self.state)
    
    def __repr__(self):
        return f"Node(state={self.state}, f={self.f_cost:.2f})"

class PlagiarismDetector:
    """
    A* based plagiarism detection system using text alignment
    """
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the plagiarism detector
        
        Args:
            similarity_threshold: Threshold for considering sentences as similar
        """
        self.similarity_threshold = similarity_threshold
    
    def preprocess_text(self, text: str, split_by_sentence: bool = True) -> List[str]:
        """
        Preprocess text into sentences or keep as paragraphs
        
        Args:
            text: Raw input text
            split_by_sentence: If True, split by sentences; if False, treat as single unit
            
        Returns:
            List of preprocessed text units
        """
        if split_by_sentence:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
        else:
            # Treat entire text as one unit (for paragraph-level comparison)
            sentences = [text]
        
        # Clean and normalize sentences
        processed_sentences = []
        for sentence in sentences:
            # Remove extra whitespace and convert to lowercase
            cleaned = sentence.strip().lower()
            # Remove punctuation except spaces
            cleaned = re.sub(r'[^\w\s]', '', cleaned)
            if cleaned:  # Only add non-empty sentences
                processed_sentences.append(cleaned)
        
        return processed_sentences
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings
        
        Args:
            s1, s2: Input strings
            
        Returns:
            Edit distance between the strings
        """
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, and substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def similarity_score(self, s1: str, s2: str) -> float:
        """
        Calculate similarity score between two strings (0 to 1)
        
        Args:
            s1, s2: Input strings
            
        Returns:
            Similarity score (1 = identical, 0 = completely different)
        """
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        
        distance = self.levenshtein_distance(s1, s2)
        return 1 - (distance / max_len)
    
    def heuristic(self, state: Tuple[int, int], doc1: List[str], doc2: List[str]) -> float:
        """
        Heuristic function for A* search
        Estimates the minimum remaining cost to reach the goal
        
        Args:
            state: Current state (i, j)
            doc1, doc2: The two documents being compared
            
        Returns:
            Estimated remaining cost
        """
        i, j = state
        remaining_doc1 = len(doc1) - i
        remaining_doc2 = len(doc2) - j
        
        # Simple heuristic: assume perfect alignment for remaining sentences
        # Cost is the minimum number of operations needed
        return abs(remaining_doc1 - remaining_doc2)
    
    def get_successors(self, state: Tuple[int, int], doc1: List[str], doc2: List[str]) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get successor states and their costs
        
        Args:
            state: Current state (i, j)
            doc1, doc2: The two documents being compared
            
        Returns:
            List of (next_state, cost) tuples
        """
        i, j = state
        successors = []
        
        # Option 1: Align current sentences from both documents
        if i < len(doc1) and j < len(doc2):
            # Cost is based on edit distance between sentences
            cost = self.levenshtein_distance(doc1[i], doc2[j])
            successors.append(((i + 1, j + 1), cost))
        
        # Option 2: Skip sentence in document 1 (deletion)
        if i < len(doc1):
            cost = len(doc1[i])  # Cost of deleting the sentence
            successors.append(((i + 1, j), cost))
        
        # Option 3: Skip sentence in document 2 (insertion)
        if j < len(doc2):
            cost = len(doc2[j])  # Cost of inserting the sentence
            successors.append(((i, j + 1), cost))
        
        return successors
    
    def astar_search(self, doc1: List[str], doc2: List[str]) -> Tuple[List[Node], float]:
        """
        A* search algorithm for text alignment
        
        Args:
            doc1, doc2: Preprocessed documents (list of sentences)
            
        Returns:
            Tuple of (path, total_cost) where path is list of nodes
        """
        # Initialize
        start_state = (0, 0)
        goal_state = (len(doc1), len(doc2))
        
        # Create start node
        start_node = Node(
            state=start_state,
            parent=None,
            g_cost=0,
            h_cost=self.heuristic(start_state, doc1, doc2)
        )
        
        # Priority queue (open list) and visited set (closed list)
        open_list = [start_node]
        heapq.heapify(open_list)
        closed_set = set()
        
        # Keep track of best g_cost for each state
        g_costs = {start_state: 0}
        
        while open_list:
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_list)
            
            # Check if goal reached
            if current_node.state == goal_state:
                # Reconstruct path
                path = []
                node = current_node
                while node:
                    path.append(node)
                    node = node.parent
                path.reverse()
                return path, current_node.g_cost
            
            # Add to closed set
            closed_set.add(current_node.state)
            
            # Explore successors
            successors = self.get_successors(current_node.state, doc1, doc2)
            
            for next_state, move_cost in successors:
                if next_state in closed_set:
                    continue
                
                # Calculate costs
                tentative_g = current_node.g_cost + move_cost
                
                # Skip if we've found a better path to this state
                if next_state in g_costs and tentative_g >= g_costs[next_state]:
                    continue
                
                # Create successor node
                h_cost = self.heuristic(next_state, doc1, doc2)
                successor_node = Node(
                    state=next_state,
                    parent=current_node,
                    g_cost=tentative_g,
                    h_cost=h_cost
                )
                
                # Update best g_cost for this state
                g_costs[next_state] = tentative_g
                
                # Add to open list
                heapq.heappush(open_list, successor_node)
        
        # No path found
        return [], float('inf')
    
    def detect_plagiarism(self, text1: str, text2: str, paragraph_mode: bool = False) -> dict:
        """
        Main function to detect plagiarism between two texts
        
        Args:
            text1, text2: Raw input texts
            paragraph_mode: If True, treat each text as single unit; if False, split by sentences
            
        Returns:
            Dictionary containing alignment results and plagiarism detection
        """
        # Preprocess texts
        doc1 = self.preprocess_text(text1, split_by_sentence=not paragraph_mode)
        doc2 = self.preprocess_text(text2, split_by_sentence=not paragraph_mode)
        
        print(f"Document 1: {len(doc1)} {'paragraph(s)' if paragraph_mode else 'sentences'}")
        print(f"Document 2: {len(doc2)} {'paragraph(s)' if paragraph_mode else 'sentences'}")
        print(f"Doc1 content: {doc1}")
        print(f"Doc2 content: {doc2}")
        
        # Run A* search for alignment
        path, total_cost = self.astar_search(doc1, doc2)
        
        if not path:
            return {"error": "No alignment found"}
        
        # Analyze alignment for plagiarism
        alignments = []
        similar_pairs = []
        
        for i in range(len(path) - 1):
            current_state = path[i].state
            next_state = path[i + 1].state
            
            # Check what type of move was made
            if (next_state[0] - current_state[0] == 1 and 
                next_state[1] - current_state[1] == 1):
                # Alignment move
                sent1_idx = current_state[0]
                sent2_idx = current_state[1]
                
                if sent1_idx < len(doc1) and sent2_idx < len(doc2):
                    sent1 = doc1[sent1_idx]
                    sent2 = doc2[sent2_idx]
                    similarity = self.similarity_score(sent1, sent2)
                    
                    alignment_info = {
                        'doc1_sentence': sent1,
                        'doc2_sentence': sent2,
                        'doc1_index': sent1_idx,
                        'doc2_index': sent2_idx,
                        'similarity': similarity,
                        'is_similar': similarity >= self.similarity_threshold
                    }
                    
                    alignments.append(alignment_info)
                    
                    if similarity >= self.similarity_threshold:
                        similar_pairs.append(alignment_info)
        
        # Calculate overall similarity
        if alignments:
            avg_similarity = sum(a['similarity'] for a in alignments) / len(alignments)
            plagiarism_score = len(similar_pairs) / len(alignments)
        else:
            avg_similarity = 0
            plagiarism_score = 0
        
        return {
            'total_cost': total_cost,
            'num_alignments': len(alignments),
            'similar_pairs': len(similar_pairs),
            'average_similarity': avg_similarity,
            'plagiarism_score': plagiarism_score,
            'alignments': alignments,
            'similar_sentences': similar_pairs
        }


"""
Test the plagiarism detection system with sample inputs
"""



if __name__ == "__main__":
    # Get the folder where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Build full paths for doc1.txt and doc2.txt
    doc1_path = os.path.join(script_dir, "doc1.txt")
    doc2_path = os.path.join(script_dir, "doc2.txt")

    # Read the files
    with open(doc1_path, "r", encoding="utf-8") as f:
        doc1 = f.read()
    with open(doc2_path, "r", encoding="utf-8") as f:
        doc2 = f.read()
    
    text1 = preprocess_document(doc1)
    text2 = preprocess_document(doc2)

    # Print results
    print("text1 =", repr(text1))
    print("text2 =", repr(text2))


    detector = PlagiarismDetector(similarity_threshold=0.6)  # Lower threshold for better detection
    
    print("=== Test Case 1: Identical Documents ===")
    result1 = detector.detect_plagiarism(text1, text2)
    print(f"Plagiarism Score: {result1['plagiarism_score']:.2f}")
    print(f"Average Similarity: {result1['average_similarity']:.2f}")
    print(f"Similar Pairs: {result1['similar_pairs']}")
    print()


