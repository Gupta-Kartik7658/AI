import heapq
from typing import List, Tuple, Optional
import re

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

# Example usage and test function
def test_plagiarism_detector():
    """
    Test the plagiarism detection system with sample inputs
    """
    detector = PlagiarismDetector(similarity_threshold=0.6)  # Lower threshold for better detection
    
    # Test Case 1: Identical documents
    text1 = """Walter Hartwell White Sr., also known by his alias Heisenberg, is a fictional character and the main protagonist of the American crime drama television series Breaking Bad. He is portrayed by Bryan Cranston.

Walter is a skilled chemist who co-founded a technology firm before he accepted a buy-out from his partners. While his partners became wealthy, Walter became a high school chemistry teacher in Albuquerque, New Mexico, barely making ends meet with his family: his wife, Skyler (Anna Gunn), and their son, Walter Jr. (RJ Mitte). At the start of the series, the day after his 50th birthday, he is diagnosed with Stage III lung cancer. After this discovery, Walter decides to manufacture and sell methamphetamine with his former student Jesse Pinkman (Aaron Paul) to ensure his family's financial security after his death. Due to his expertise, Walter's "blue meth" is purer than any other on the market, and he is pulled deeper into the illicit drug trade.

An antihero[a] turned villain protagonist as the series progresses, Walter becomes increasingly ruthless and unsympathetic, as the series' creator, Vince Gilligan, wanted him to turn from "Mr. Chips into Scarface". He adopts the alias "Heisenberg", which becomes recognizable as a kingpin figure in the Southwestern drug trade. Walter struggles with managing his family while hiding his involvement in the drug business from his brother-in-law, Hank Schrader (Dean Norris), an agent of the Drug Enforcement Administration. Although AMC officials initially hesitated to cast Cranston due to his previous comedic role in Malcolm in the Middle, Gilligan cast him based on his past performance in The X-Files episode "Drive", which Gilligan wrote. Cranston contributed greatly to the creation of his character, including Walter's backstory, personality, and physical appearance.

Both Walter and Cranston's performance have received critical acclaim, and Walter has frequently been mentioned as one of the greatest and most iconic television characters ever created. Cranston won four Primetime Emmy Awards for Outstanding Lead Actor in a Drama Series, three of them being consecutive. He is the first man to win a Critics' Choice, Golden Globe, Primetime Emmy, and Screen Actors Guild Award for his performance. Cranston reprised the role in a flashback for Breaking Bad's sequel film, El Camino: A Breaking Bad Movie, and again in the sixth and final season of the prequel series Better Call Saul, making him one of the few characters to appear in all three, alongside Jesse Pinkman, Mike Ehrmantraut (Jonathan Banks), Ed Galbraith (Robert Forster), and Austin Ramey (Todd Terry). """
    text2 = """Walter Hartwell White Sr., better known under the alias Heisenberg, is the central figure of the American crime drama Breaking Bad. The role is portrayed by actor Bryan Cranston.

Once a brilliant chemist and co-founder of a technology startup, Walter walked away after accepting a buyout from his partners, who went on to achieve great success. In contrast, he found himself working as a high school chemistry teacher in Albuquerque, New Mexico, supporting his wife Skyler (Anna Gunn) and their teenage son, Walter Jr. (RJ Mitte), while struggling financially. On the day after his 50th birthday, Walter is diagnosed with Stage III lung cancer. Determined to leave behind financial security for his family, he partners with his former student Jesse Pinkman (Aaron Paul) to produce and sell methamphetamine. Thanks to his scientific skill, his distinctive blue product proves unmatched in purity, drawing him deeper into the criminal underworld.

Over time, Walter shifts from reluctant antihero to ruthless villain, fulfilling creator Vince Gilligan’s vision of transforming him from “Mr. Chips into Scarface.” He adopts the moniker Heisenberg, which becomes infamous across the Southwest drug trade. As his empire grows, Walter faces mounting tension at home and must conceal his secret life from his brother-in-law Hank Schrader (Dean Norris), an agent with the DEA. Casting Cranston was initially questioned by AMC executives due to his comedic history in Malcolm in the Middle, but Gilligan fought for him based on his performance in the X-Files episode “Drive.” Cranston further helped shape Walter’s persona, contributing ideas about his backstory, mannerisms, and appearance.

Both the character and Cranston’s portrayal have been widely praised, with Walter often ranked among television’s most iconic figures. Cranston earned four Primetime Emmy Awards for Outstanding Lead Actor in a Drama Series, including three in a row, and became the first male actor to win an Emmy, Golden Globe, Critics’ Choice, and Screen Actors Guild Award for a single role. He reprised Walter White in a flashback for the sequel film El Camino: A Breaking Bad Movie, as well as in the final season of the prequel series Better Call Saul. Alongside Jesse Pinkman, Mike Ehrmantraut (Jonathan Banks), Ed Galbraith (Robert Forster), and Austin Ramey (Todd Terry), Walter is one of the rare characters to appear in all three productions."""
    
    print("=== Test Case 1: Identical Documents ===")
    result1 = detector.detect_plagiarism(text1, text2)
    print(f"Plagiarism Score: {result1['plagiarism_score']:.2f}")
    print(f"Average Similarity: {result1['average_similarity']:.2f}")
    print(f"Similar Pairs: {result1['similar_pairs']}")
    print()
    
    # # Test Case 2: Your specific test case (paragraph mode)
    # doc1_orig = "Artificial intelligence has become a transformative force in modern society. From healthcare to finance, AI systems are reshaping how humans interact with technology and make decisions."
    # doc2_similar = "Artificial intelligence is a powerful force in today's society. In fields such as healthcare and finance, AI is changing the way people interact with technology and the way decisions are made."
    
    # print("=== Test Case 2: Your Specific Example (Paragraph Mode) ===")
    # result2 = detector.detect_plagiarism(doc1_orig, doc2_similar, paragraph_mode=True)
    # print(f"Plagiarism Score: {result2['plagiarism_score']:.2f}")
    # print(f"Average Similarity: {result2['average_similarity']:.2f}")
    # print(f"Similar Pairs: {result2['similar_pairs']}")
    
    # # Print detailed alignment info
    # if result2['alignments']:
    #     for alignment in result2['alignments']:
    #         print(f"Similarity: {alignment['similarity']:.3f}")
    #         print(f"Doc1: {alignment['doc1_sentence'][:100]}...")
    #         print(f"Doc2: {alignment['doc2_sentence'][:100]}...")
    #         print(f"Is Similar: {alignment['is_similar']}")
    # print()
    
    # # Test Case 3: Your specific test case (sentence mode)
    # print("=== Test Case 3: Your Specific Example (Sentence Mode) ===")
    # result3 = detector.detect_plagiarism(doc1_orig, doc2_similar, paragraph_mode=False)
    # print(f"Plagiarism Score: {result3['plagiarism_score']:.2f}")
    # print(f"Average Similarity: {result3['average_similarity']:.2f}")
    # print(f"Similar Pairs: {result3['similar_pairs']}")
    # print()
    
    # # Test Case 4: Completely different documents
    # text5 = "The weather is sunny today. I like to read books. Programming is interesting."
    # text6 = "Mathematics is challenging. Sports are fun to watch. Music helps me relax."
    
    # print("=== Test Case 4: Completely Different Documents ===")
    # result4 = detector.detect_plagiarism(text5, text6)
    # print(f"Plagiarism Score: {result4['plagiarism_score']:.2f}")
    # print(f"Average Similarity: {result4['average_similarity']:.2f}")
    # print(f"Similar Pairs: {result4['similar_pairs']}")
    
    return detector

if __name__ == "__main__":
    # Run the test
    detector = test_plagiarism_detector()