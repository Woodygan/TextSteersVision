import requests
import json
import os
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from sae_lens import SAE  # pip install sae-lens
import pdb 
def get_neuronpedia_urls(model_name, layer_idx, feature_ids):
    """
    Generate Neuronpedia URLs for the given features.
    
    Args:
        model_name: Name of the model
        layer_idx: Layer index
        feature_ids: List of feature IDs
        
    Returns:
        Dictionary mapping feature IDs to their Neuronpedia URLs
    """
    urls = {}
    
    if "gemma-2-9b" in model_name:
        base_url = f"https://www.neuronpedia.org/api/feature/gemma-2-9b/{layer_idx}-gemmascope-res-16k"
        for feature_id in feature_ids:
            urls[feature_id] = f"{base_url}/{feature_id}"
    elif "gemma-2-2b" in model_name:
        base_url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer_idx}-gemmascope-res-16k"
        for feature_id in feature_ids:
            urls[feature_id] = f"{base_url}/{feature_id}"
    elif "Llama-3.1-8B" in model_name:
        base_url = f"https://www.neuronpedia.org/api/feature/llama3.1-8b/{layer_idx}-llamascope-res-32k"  
        for feature_id in feature_ids:
            urls[feature_id] = f"{base_url}/{feature_id}"
    
    return urls

def fetch_feature(url):
    """
    Fetch feature data from Neuronpedia API and save it to a JSON file.

    Args:
        url: The Neuronpedia API URL.
    Returns:
        A dictionary with keys 'explanation' and 'top_activations', or False on error.
    """
    try:
        print(f"Fetching data from: {url}")
        response = requests.get(url, timeout=10)

        # Check if request was successful
        if response.status_code == 200:
            try:
                # Parse JSON response
                response_json = response.json()
                extracted_info = {} 
                
                # Extract only the explanation  
                extracted_info['explanation'] = response_json['explanations'][0]['description']
                
                # Extract only the top activation tokens
                top_activations = []
                for activation in response_json['activations'][:30]:  # Limit to top 30 for conciseness
                    if 'tokens' in activation and 'values' in activation:
                        tokens = activation.get('tokens', [])
                        values = activation.get('values', [])
                        
                        # Find top 3 token indices based on values
                        if len(tokens) == len(values) and len(tokens) > 0:
                            # Create (value, index) pairs and sort them
                            token_value_pairs = [(val, idx) for idx, val in enumerate(values)]
                            token_value_pairs.sort(reverse=True)  # Sort by value in descending order
                            
                            # Get indices of top 3 (or fewer if there aren't 3) tokens
                            top_indices = set([idx for _, idx in token_value_pairs[:min(3, len(token_value_pairs))]])
                            
                            # Clean tokens and highlight top 3
                            clean_tokens = []
                            for i, token in enumerate(tokens):
                                # Remove the \u2581 token (it's a special character used for whitespace in some tokenizers)
                                clean_token = token.replace('\u2581', '')
                                if clean_token:  # Only include non-empty tokens
                                    if i in top_indices:
                                        clean_tokens.append(f"<top>{clean_token}</top>")
                                    else:
                                        clean_tokens.append(clean_token)
                            
                            # Join tokens to form readable text
                            joined_text = ' '.join(clean_tokens) if clean_tokens else "No tokens available"
                            top_activations.append(joined_text)
                        else:
                            # Fallback if values and tokens don't match
                            clean_tokens = [token.replace('\u2581', '') for token in tokens]
                            joined_text = ' '.join([token for token in clean_tokens if token])
                            top_activations.append(joined_text if joined_text else "No tokens available")
                    else:
                        # No tokens or values available
                        top_activations.append("No tokens available")
                
                # Deduplicate top activations (remove duplicates or very similar texts)
                seen_texts = set()
                unique_activations = []
                for text in top_activations:
                    # Convert to lowercase for case-insensitive comparison, but remove tags for comparison
                    text_for_comparison = text.lower().replace("<top>", "").replace("</top>", "")
                    
                    # Skip if exact match or very similar text already exists
                    if text_for_comparison in seen_texts:
                        continue
                    
                    # Check for high similarity with existing texts
                    similar_exists = False
                    for seen_text in seen_texts:
                        # Simple similarity check - if one is a substring of the other
                        if text_for_comparison in seen_text or seen_text in text_for_comparison:
                            similar_exists = True
                            break
                    
                    if not similar_exists:
                        seen_texts.add(text_for_comparison)
                        unique_activations.append(text)

                top_activations = unique_activations[:10]  # Ensure we still limit to 10 examples
                extracted_info['top_activations'] = top_activations
                return extracted_info

            except json.JSONDecodeError:
                print("Error: Response was not valid JSON")
                print(f"Response content: {response.text[:200]}...")
                return False
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            return False

    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def fetch_features(model_name, layer_idx, feature_ids):
    """
    Fetch feature data from Neuronpedia API and save it to a JSON file.

    Args:
        model_name: Name of the model
        layer_idx: Layer index
        feature_ids: List of feature IDs
    Returns:
        A dictionary mapping feature IDs to their extracted information
    """
    urls = get_neuronpedia_urls(model_name, layer_idx, feature_ids)
    features = {}
    
    for feature_id, url in urls.items():
        feature_data = fetch_feature(url)
        if feature_data:
            features[feature_id] = feature_data
    
    return features

def get_taxonomies():
    """
    Returns a dictionary of taxonomies relevant to vision-language models.
    
    Returns:
        Dictionary with taxonomy names as keys and descriptions as values
    """
    return {
        "spatial_relationship": "Words describing relative positions (e.g., 'on', 'under', 'beside')",
        "counting": "Words or phrases indicating quantities (e.g., 'three', 'several', 'few')",
        "attribute": "Words describing characteristics (e.g., 'red', 'large', 'smooth')",
        "entity": "Objects or subjects in the scene (e.g., 'cat', 'table', 'person')",
        "action": "Verbs describing activities (e.g., 'running', 'eating', 'holding')",
        "temporal": "Time-related concepts (e.g., 'before', 'after', 'during')",
        "comparison": "Words comparing entities (e.g., 'larger', 'fewer', 'same')",
        "negation": "Negation words (e.g., 'not', 'no', 'never')"
    }

def generate_taxonomy_sentences(taxonomy: str) -> List[Tuple[str, str]]:
    """
    Generate 20 diverse example sentences for a given taxonomy with highlighted tokens.
    Each highlighted token is guaranteed to be a single token.
    
    Args:
        taxonomy: The taxonomy name
        
    Returns:
        List of tuples (sentence, highlighted_token)
    """
    sentences = {
        "spatial_relationship": [
            ("The cat is on the table", "on"),
            ("She put the book under the chair", "under"),
            ("The plant grows beside the window", "beside"),
            ("The lamp hangs above the desk", "above"),
            ("The keys are inside the drawer", "inside"),
            ("The dog sits between two trees", "between"),
            ("The car parked behind the house", "behind"),
            ("The child stands near the fountain", "near"),
            ("He placed the shoes beneath the bed", "beneath"),
            ("The painting hangs over the fireplace", "over"),
            ("The toy fell off the shelf", "off"),
            ("The boat floats along the river", "along"),
            ("The fence runs around the yard", "around"),
            ("The path leads toward the mountain", "toward"),
            ("The restaurant is opposite the bank", "opposite"),
            ("The statue stands amid the garden", "amid"),
            ("She stood to the left of the store", "left"),
            ("The pen rolled down the table", "down"),
            ("The cat is to the right of the chair", "right"),
            ("The car drove through the tunnel", "through")
        ],
        "counting": [
            ("There are three apples in the basket", "three"),
            ("The teacher counted five children", "five"),
            ("They bought a dozen eggs", "dozen"),
            ("I need two cups of sugar", "two"),
            ("There is one glass on the table", "one"),
            ("The recipe requires four eggs", "four"),
            ("He made zero mistakes on the test", "zero"),
            ("She won seven medals in the competition", "seven"),
            ("I can see six birds on the wire", "six"),
            ("The box contains eight pencils", "eight"),
            ("She ate nine grapes for breakfast", "nine"),
            ("We need ten volunteers for the event", "ten"),
            ("The package has eleven items inside", "eleven"),
            ("There are twelve months in a year", "twelve"),
            ("I bought fifteen apples at the market", "fifteen"),
            ("The clock showed twenty minutes past nine", "twenty"),
            ("The team scored thirty points in the game", "thirty"),
            ("This bottle holds fifty milliliters of water", "fifty"),
            ("The book has forty chapters in total", "forty"),
            ("About sixty people attended the meeting", "sixty")
        ],
        "attribute": [
            ("The red car stopped at the light", "red"),
            ("She wore a beautiful dress", "beautiful"),
            ("The heavy box fell to the floor", "heavy"),
            ("A delicious meal was prepared", "delicious"),
            ("The smooth surface reflected light", "smooth"),
            ("A loud noise woke everyone up", "loud"),
            ("The ancient ruins fascinated tourists", "ancient"),
            ("The tiny insect landed on the leaf", "tiny"),
            ("A bright light illuminated the room", "bright"),
            ("The tall building cast a shadow", "tall"),
            ("Her soft voice calmed the child", "soft"),
            ("The sharp knife cut through the meat", "sharp"),
            ("The deep lake contains many fish", "deep"),
            ("A sour taste lingered in his mouth", "sour"),
            ("The narrow path led to the summit", "narrow"),
            ("The cold water refreshed the hikers", "cold"),
            ("An empty bottle rolled on the floor", "empty"),
            ("The young tree bent in the wind", "young"),
            ("His fierce expression frightened them", "fierce"),
            ("The sweet fragrance filled the room", "sweet")
        ],
        "entity": [
            ("The dog barked at the mailman", "dog"),
            ("A tree fell during the storm", "tree"),
            ("The computer crashed unexpectedly", "computer"),
            ("The child played in the park", "child"),
            ("The ocean waves crashed on rocks", "ocean"),
            ("The moon shone brightly at night", "moon"),
            ("A bird built a nest in the garden", "bird"),
            ("The mountain peak was snow covered", "mountain"),
            ("The phone rang in the night", "phone"),
            ("The cake was delicious and moist", "cake"),
            ("The car stopped at the light", "car"),
            ("A book sat on the desk", "book"),
            ("The river flooded the village", "river"),
            ("A cat chased the mouse", "cat"),
            ("The sun rose over the horizon", "sun"),
            ("Her house was painted blue", "house"),
            ("The plane landed safely", "plane"),
            ("The flower bloomed in spring", "flower"),
            ("The teacher explained the lesson", "teacher"),
            ("The key unlocked the door", "key")
        ],
        "action": [
            ("The athlete is running in the marathon", "running"),
            ("She is writing a novel", "writing"),
            ("The chef is cooking dinner", "cooking"),
            ("They are building a new house", "building"),
            ("The baby is sleeping peacefully", "sleeping"),
            ("The audience is clapping enthusiastically", "clapping"),
            ("The cat is chasing a mouse", "chasing"),
            ("He is driving to work", "driving"),
            ("The teacher is explaining the lesson", "explaining"),
            ("They are singing a beautiful song", "singing"),
            ("She is dancing gracefully", "dancing"),
            ("He is reading a book", "reading"),
            ("The dog is barking loudly", "barking"),
            ("The students are studying for exams", "studying"),
            ("They are walking along the beach", "walking"),
            ("She is painting a landscape", "painting"),
            ("The birds are flying south", "flying"),
            ("He is swimming in the lake", "swimming"),
            ("She is laughing at the joke", "laughing"),
            ("They are eating dinner together", "eating")
        ],
        "temporal": [
            ("I will call you after the meeting", "after"),
            ("Before sunrise they began hiking", "Before"),
            ("During the concert it started raining", "During"),
            ("She arrived while we were eating", "while"),
            ("Until tomorrow we won't know", "Until"),
            ("Since last year things have changed", "Since"),
            ("We waited until the storm passed", "until"),
            ("He was nervous before his speech", "before"),
            ("Following the ceremony was a reception", "Following"),
            ("She completed her work subsequently", "subsequently"),
            ("She always arrives early", "always"),
            ("They never visit on Sundays", "never"),
            ("He sometimes forgets his keys", "sometimes"),
            ("The shop opens daily at nine", "daily"),
            ("They meet weekly to discuss progress", "weekly"),
            ("The train arrives hourly", "hourly"),
            ("Soon the winter will arrive", "Soon"),
            ("Later we can discuss that issue", "Later"),
            ("Currently we face many challenges", "Currently"),
            ("Eventually everyone left the party", "Eventually")
        ],
        "comparison": [
            ("This book is better than that one", "better"),
            ("She runs faster than her brother", "faster"),
            ("The blue car is more expensive", "more"),
            ("This is the largest building in town", "largest"),
            ("He scored higher on the test", "higher"),
            ("The weather is worse today", "worse"),
            ("These shoes are cheaper than those", "cheaper"),
            ("This is the most beautiful painting", "most"),
            ("The new phone is smaller but powerful", "smaller"),
            ("Today's assignment was easier", "easier"),
            ("She is taller than her sister", "taller"),
            ("This coffee is stronger than usual", "stronger"),
            ("His car is newer than mine", "newer"),
            ("Her voice is louder than his", "louder"),
            ("This method is simpler to understand", "simpler"),
            ("That explanation was clearer", "clearer"),
            ("The blue team performed best", "best"),
            ("This soup tastes sweeter", "sweeter"),
            ("The older building looks nicer", "nicer"),
            ("Their house is bigger than ours", "bigger")
        ],
        "negation": [
            ("He did not attend the meeting", "not"),
            ("There is no water in the glass", "no"),
            ("She never eats meat", "never"),
            ("They could not find their keys", "not"),
            ("It is impossible to finish on time", "impossible"),
            ("The store is nowhere near here", "nowhere"),
            ("Nothing happened after we left", "Nothing"),
            ("I barely slept last night", "barely"),
            ("They hardly recognized their friend", "hardly"),
            ("The package has not arrived yet", "not"),
            ("Nobody came to the party", "Nobody"),
            ("He refuses to cooperate with us", "refuses"),
            ("I disagree with your opinion", "disagree"),
            ("She denied breaking the window", "denied"),
            ("They rejected our proposal", "rejected"),
            ("The claim lacks supporting evidence", "lacks"),
            ("She failed the examination", "failed"),
            ("The project was unsuccessful", "unsuccessful"),
            ("He forbids smoking in his house", "forbids"),
            ("I cannot remember her name", "cannot")
        ]
    }
    
    return sentences.get(taxonomy, [])

def analyze_with_gemmascope(model_name: str, sentence: str, highlighted_token: str, 
                           layers: List[int], tokenizer=None, model=None) -> Dict[int, List[Tuple[int, int]]]:
    """
    Use GemmaScope to identify features highly activated on the highlighted token.
    
    Args:
        model_name: Name of the model to use
        sentence: The sentence to analyze
        highlighted_token: The token to focus on
        layers: List of layer indices to analyze
        tokenizer: Pre-loaded tokenizer (optional)
        model: Pre-loaded model (optional)
        
    Returns:
        Dictionary mapping layer indices to lists of (feature_id, rank) tuples
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Determine model configuration
        if "gemma-2-9b" in model_name or "10b" in model_name:
            model_id = "google/gemma-2-9b"
            release = "gemma-scope-9b-pt-res-canonical"
            width = "16k"
        elif "gemma-2-2b" in model_name:
            model_id = "google/gemma-2-2b"
            release = "gemma-scope-2b-pt-res-canonical"
            width = "16k"
        elif "Llama-3.1-8B" in model_name:
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
            release = "llama_scope_lxr_8x"
            width = "32k"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Use provided tokenizer and model or load them
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(model_id)
        
        # Check if CUDA is available and model is on GPU
        device = next(model.parameters()).device
        
        # Tokenize the sentence
        tokens = tokenizer.tokenize(sentence)
        token_ids = tokenizer.encode(sentence, add_special_tokens=True)
        
        # Find the token position for the highlighted token
        highlighted_positions = []
        for i, token in enumerate(tokens):
            # Check if this token matches the highlighted token
            if highlighted_token.lower() in token.lower():
                # Add 1 to account for the special token at the beginning ([BOS])
                highlighted_positions.append(i+1)
        
        if not highlighted_positions:
            print(f"WARNING: Could not find highlighted token '{highlighted_token}' in tokenized sentence.")
            print(f"Tokenized sentence: {tokens}")
            return {}
        
        # Run the model to get hidden states (only once per sentence)
        # Ensure input_ids is on the same device as the model
        input_ids = torch.tensor([token_ids]).to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
        
        # Process each layer
        result = {}
        for layer_idx in layers:
            try:
                if "gemma" in model_name.lower():
                    sae_id = f"layer_{layer_idx}/width_{width}/canonical"
                elif "llama" in model_name.lower():
                    sae_id =f"l{layer_idx}r_8x"
                else:
                    raise ValueError(f"Unsupported model: {model_name}")
                # Load the SAE model for this layer with sparsity information
                sae, cfg_dict, sparsity = SAE.from_pretrained(
                    release=release,
                    sae_id=sae_id,
                )
                
                # Get the hidden states for this layer and move to CPU if needed
                # SAE might expect CPU inputs
                layer_hidden_states = hidden_states[layer_idx].cpu()
                
                # Use the SAE to encode the hidden states
                sae_acts = sae.encode(layer_hidden_states)
                
                # Get activations for the highlighted token positions
                token_activations = []
                for pos in highlighted_positions:
                    if pos < sae_acts.shape[1]:  # Check if position is within bounds
                        token_activations.append(sae_acts[0, pos, :])
                
                if not token_activations:
                    print(f"WARNING: Token positions {highlighted_positions} are out of bounds for activations shape {sae_acts.shape}")
                    continue
                
                # Find the token occurrence with the maximum activation for each feature
                if len(token_activations) == 1:
                    # If there's only one occurrence, use that
                    max_activation = token_activations[0]
                else:
                    # If there are multiple occurrences, find the maximum activation for each feature
                    stacked_activations = torch.stack(token_activations)
                    max_activation, _ = torch.max(stacked_activations, dim=0)
                
                # Get the top 20 activated features, filtering out garbage features if sparsity info is available
                if sparsity is not None:
                    # Get the log sparsity for each feature - lower values indicate higher quality features
                    # Only consider features with reasonable sparsity (e.g., sparsity < 0)
                    valid_features_mask = (sparsity < 0)
                    filtered_activation = max_activation.clone()
                    # Set garbage feature activations to -inf to exclude them from topk
                    filtered_activation[~valid_features_mask] = float('-inf')
                    top_features = torch.topk(filtered_activation, min(20, torch.sum(valid_features_mask).item()))
                    print(f"Layer {layer_idx}: Filtered out {torch.sum(~valid_features_mask).item()} garbage features using sparsity info")
                else:
                    # No sparsity info available, use all features
                    top_features = torch.topk(max_activation, min(20, max_activation.shape[0]))
                
                # Store feature IDs with their ranks (starting from 1)
                feature_ids_with_ranks = [(int(idx), rank+1) for rank, idx in enumerate(top_features.indices.tolist())]
                
                result[layer_idx] = feature_ids_with_ranks
                print(f"Layer {layer_idx}: Found {len(feature_ids_with_ranks)} activated features for token '{highlighted_token}'")
                
            except Exception as e:
                print(f"Error processing layer {layer_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        return result
    
    except Exception as e:
        print(f"Error initializing GemmaScope analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to simulated results for testing
        import random
        result = {}
        
        return result

def find_common_features(feature_sets: List[Set[int]]) -> Set[int]:
    """
    Find the intersection of multiple sets of feature IDs.
    
    Args:
        feature_sets: List of sets, each containing feature IDs
        
    Returns:
        Set of common feature IDs
    """
    if not feature_sets:
        return set()
    
    # Start with the first set and find intersection with all others
    common = feature_sets[0]
    for feature_set in feature_sets[1:]:
        common = common.intersection(feature_set)
    
    return common

def count_common_features(feature_sets: List[Set[int]]) -> Dict[int, int]:
    """
    Count occurrences of each feature across the feature sets.
    
    Args:
        feature_sets: List of sets, each containing feature IDs
        
    Returns:
        Dictionary mapping feature IDs to their occurrence counts
    """
    if not feature_sets:
        return {}
    
    # Count occurrences of each feature
    feature_counts = {}
    for feature_set in feature_sets:
        for feature_id in feature_set:
            if feature_id in feature_counts:
                feature_counts[feature_id] += 1
            else:
                feature_counts[feature_id] = 1
    
    return feature_counts

def count_weighted_features(feature_sets_with_ranks: List[List[Tuple[int, int]]]) -> Dict[int, float]:
    """
    Count occurrences of each feature across the feature sets, with weighting based on rank.
    
    Args:
        feature_sets_with_ranks: List of lists, each containing (feature_id, rank) tuples
        
    Returns:
        Dictionary mapping feature IDs to their weighted scores
    """
    if not feature_sets_with_ranks:
        return {}
    
    # Count occurrences and weighted scores of each feature
    feature_scores = {}
    
    for features_with_ranks in feature_sets_with_ranks:
        # Normalize weights based on max rank (usually 20)
        max_rank = max([rank for _, rank in features_with_ranks]) if features_with_ranks else 1
        
        for feature_id, rank in features_with_ranks:
            # Weight is inverse of normalized rank (1.0 for rank 1, decreasing as rank increases)
            weight = 1.0 - ((rank - 1) / max_rank)
            
            if feature_id in feature_scores:
                # Add occurrence count and weighted score
                feature_scores[feature_id]['count'] += 1
                feature_scores[feature_id]['weighted_score'] += weight
            else:
                feature_scores[feature_id] = {
                    'count': 1,
                    'weighted_score': weight
                }
    
    return feature_scores

def get_top_features(feature_counts: Dict[int, int], top_n: int = 32) -> List[int]:
    """
    Get the top N most frequent features.
    
    Args:
        feature_counts: Dictionary mapping feature IDs to their counts
        top_n: Number of top features to return
        
    Returns:
        List of feature IDs for the top N most frequent features
    """
    # Sort features by count (descending) and return top N
    sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
    return [feature_id for feature_id, _ in sorted_features[:top_n]]

def get_top_weighted_features(feature_scores: Dict[int, Dict], top_n: int = 10, min_count: int = 2) -> List[int]:
    """
    Get the top N features based on weighted scores, with minimum occurrence requirement.
    
    Args:
        feature_scores: Dictionary mapping feature IDs to score dictionaries
        top_n: Number of top features to return
        min_count: Minimum number of occurrences required
        
    Returns:
        List of feature IDs for the top N features by weighted score
    """
    # Filter features by minimum count
    filtered_features = {
        feature_id: scores 
        for feature_id, scores in feature_scores.items() 
        if scores['count'] >= min_count
    }
    
    # Sort features by weighted score (descending)
    sorted_features = sorted(
        filtered_features.items(), 
        key=lambda x: x[1]['weighted_score'], 
        reverse=True
    )
    
    # Return top N feature IDs
    return [feature_id for feature_id, _ in sorted_features[:top_n]]

def check_feature_alignment(feature_data: dict, taxonomy: str) -> bool:
    """
    Check if a feature aligns with a taxonomy by querying GPT-o3-mini.
    
    Args:
        feature_data: Dictionary containing feature explanation and top activations
        taxonomy: The taxonomy to check alignment with
        
    Returns:
        Boolean indicating whether the feature aligns with the taxonomy
    """
    try:
        import requests
        
        # Get definitions and examples for the taxonomy
        taxonomy_definitions = {
            "spatial_relationship": "Words describing relative positions in space (e.g., 'on', 'under', 'beside', 'above', 'below', 'between', 'inside')",
            "counting": "Words or phrases indicating specific quantities or numbers (e.g., 'three', 'five', 'dozen', 'two', 'one', 'four', 'zero', 'seven')",
            "attribute": "Words describing characteristics or qualities of objects or entities (e.g., 'red', 'beautiful', 'heavy', 'smooth', 'loud', 'bright', 'tall')",
            "entity": "Objects or subjects that can be identified in scenes or situations (e.g., 'dog', 'tree', 'computer', 'child', 'car', 'book')",
            "action": "Verbs describing activities or things that can be done (e.g., 'running', 'writing', 'cooking', 'building', 'sleeping', 'driving')",
            "temporal": "Time-related concepts or words indicating when events happen (e.g., 'after', 'before', 'during', 'while', 'until', 'always', 'never')",
            "comparison": "Words comparing entities in terms of their attributes (e.g., 'better', 'faster', 'more', 'largest', 'higher', 'smaller', 'louder')",
            "negation": "Words expressing denial, refusal, or indicating the absence of something (e.g., 'not', 'no', 'never', 'impossible', 'nowhere', 'nothing')"
        }
        
        # Example features that align with each taxonomy (for few-shot learning)
        # Update with highlighted keywords for better demonstration
        taxonomy_examples = {
            "spatial_relationship": [
                {"explanation": "This feature detects tokens representing objects placed on top of surfaces", 
                 "activations": ["The book is <top>on</top> the table", "A cup sitting <top>on</top> the counter", "Papers <top>on</top> the desk"]},
                {"explanation": "This feature activates on phrases involving things contained inside other things", 
                 "activations": ["Documents <top>inside</top> the folder", "A toy <top>inside</top> the box", "The key is <top>inside</top> the drawer"]}
            ],
            "counting": [
                {"explanation": "This feature responds to mentions of small numerical quantities", 
                 "activations": ["There are <top>three</top> apples", "The team of <top>five</top> people", "<top>Two</top> cats sleeping"]},
                {"explanation": "This feature activates on tokens referring to counting or enumeration", 
                 "activations": ["<top>Eight</top> different options", "First, second, and <top>third</top> place", "Counted <top>twenty</top> sheep"]}
            ],
            "attribute": [
                {"explanation": "This feature detects descriptions of color attributes", 
                 "activations": ["The <top>red</top> dress", "A <top>blue</top> sky", "<top>Green</top> leaves on the trees"]},
                {"explanation": "This feature recognizes words describing texture qualities", 
                 "activations": ["The <top>smooth</top> surface", "<top>Rough</top> edges", "A <top>soft</top> pillow"]}
            ],
            "entity": [
                {"explanation": "This feature activates on references to animal entities", 
                 "activations": ["The <top>dog</top> barked", "A <top>cat</top> jumped", "<top>Birds</top> flying overhead"]},
                {"explanation": "This feature responds to mentions of vehicles or transportation", 
                 "activations": ["The <top>car</top> stopped", "A <top>plane</top> in the sky", "Riding the <top>train</top>"]}
            ],
            "action": [
                {"explanation": "This feature detects verbs describing movement", 
                 "activations": ["She was <top>running</top>", "The child <top>walking</top> slowly", "They <top>jumped</top> over the fence"]},
                {"explanation": "This feature recognizes cooking or food preparation actions", 
                 "activations": ["<top>Cooking</top> dinner", "She <top>baked</top> cookies", "He was <top>chopping</top> vegetables"]}
            ],
            "temporal": [
                {"explanation": "This feature activates on words indicating sequence or order in time", 
                 "activations": ["<top>After</top> the game", "<top>Before</top> sunrise", "<top>During</top> the meeting"]},
                {"explanation": "This feature detects adverbs of frequency", 
                 "activations": ["<top>Always</top> on time", "<top>Never</top> late", "<top>Sometimes</top> forgets"]}
            ],
            "comparison": [
                {"explanation": "This feature responds to comparative adjectives", 
                 "activations": ["<top>Better</top> than expected", "<top>Faster</top> than light", "<top>Larger</top> than average"]},
                {"explanation": "This feature detects superlative forms", 
                 "activations": ["The <top>best</top> result", "<top>Highest</top> mountain", "<top>Smallest</top> particle"]}
            ],
            "negation": [
                {"explanation": "This feature activates on explicit negation words", 
                 "activations": ["<top>Not</top> going", "<top>No</top> entry", "<top>Never</top> again"]},
                {"explanation": "This feature detects phrases expressing absence or lack", 
                 "activations": ["<top>Nothing</top> remains", "<top>Nobody</top> came", "<top>Nowhere</top> to be found"]}
            ]
        }
        
        # Create prompt with feature information and taxonomy definition
        prompt = f"""Task: Determine if a neural network's sparse autoencoder (SAE) feature aligns with the taxonomy "{taxonomy}".

Taxonomy Definition: {taxonomy_definitions.get(taxonomy, f"The {taxonomy} taxonomy")}

Feature Information:
1. Feature's explanation: {feature_data['explanation']}
2. Top activation examples (tokens wrapped in <top>...</top> have the highest activation values and are the most important to focus on):
"""
        
        # Add activation examples
        for i, example in enumerate(feature_data.get('top_activations', [])[:5]):
            prompt += f"   {i+1}. {example}\n"
        
        # Add few-shot examples for the specified taxonomy
        prompt += f"\nExamples of features that DO align with the {taxonomy} taxonomy (notice how the key words are highlighted with <top>...</top> tags):\n"
        for i, example in enumerate(taxonomy_examples.get(taxonomy, [])[:2]):
            prompt += f"Example {i+1}:\n"
            prompt += f"- Explanation: {example['explanation']}\n"
            prompt += f"- Activations: {', '.join(example['activations'][:3])}\n"
        
        # Add the binary question with emphasis on focusing on highlighted tokens
        prompt += f"""
When making your decision, you should follow these rules:
1. First pay attention to the feature's explanation.
2. If you cannot decice, you should then pay special attention to the tokens highlighted with <top>...</top> tags, as these are the most highly activated tokens and strongest indicators of what the feature detects.
3. Also consider the diversity of the activation examples provided. If one feature only activates one particular work, it may not be as aligned as a feature that activates on a variety of words. 

Based on the feature's explanation and the highlighted tokens in the activation examples, does this feature specifically detect or respond to {taxonomy_definitions.get(taxonomy, taxonomy)}? Your answer should start with YES or NO, then provide a brief reason. Do not start with any other words or phrases such as `answer`."""
        
        api_url = "https://api.openai.com/v1/chat/completions"
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            print("ERROR!!!: OPENAI_API_KEY environment variable not set. Skipping alignment check.")
            return False  # Default to excluding the feature if API key is not available
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": "o3-mini-2025-01-31",
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        
        response = requests.post(api_url, headers=headers, json=data)
        
        if response.status_code == 200:
            response_json = response.json()
            answer = response_json['choices'][0]['message']['content']
            # Parse the answer to determine if it's a YES
            answer_filtered = answer.strip().lower()
            answer_filtered = answer_filtered.replace("answer:", "")
            alignment = answer_filtered.strip().startswith("yes")
            
            print(f"Alignment check for taxonomy '{taxonomy}':")
            print(f"  Feature explanation: {feature_data['explanation'][:100]}...")
            print(f"  Result: {'ALIGNED' if alignment else 'NOT ALIGNED'}")
            print(f"  Reason: {answer[:200]}...")
            
            return alignment
        else:
            print(f"API request failed with status code {response.status_code}: {response.text}")
            return True  # Default to including the feature if API request fails
            
    except Exception as e:
        print(f"Error checking feature alignment: {str(e)}")
        import traceback
        traceback.print_exc()
        return True  # Default to including the feature if there's an error

def get_union_features(feature_sets_with_ranks: List[List[Tuple[int, int]]]) -> List[int]:
    """
    Get the union of all feature IDs across all sentences.
    
    Args:
        feature_sets_with_ranks: List of lists, each containing (feature_id, rank) tuples
        
    Returns:
        List of unique feature IDs from all sets
    """
    if not feature_sets_with_ranks:
        return []
    
    # Extract all feature IDs
    all_features = set()
    for features_with_ranks in feature_sets_with_ranks:
        for feature_id, _ in features_with_ranks:
            all_features.add(feature_id)
    
    return list(all_features)

def get_intersection_features(feature_sets_with_ranks: List[List[Tuple[int, int]]], min_occurrences: int = 2) -> List[int]:
    """
    Get features that appear in at least min_occurrences sets.
    
    Args:
        feature_sets_with_ranks: List of lists, each containing (feature_id, rank) tuples
        min_occurrences: Minimum number of occurrences required
        
    Returns:
        List of feature IDs that appear in at least min_occurrences sets
    """
    if not feature_sets_with_ranks or min_occurrences < 1:
        return []
    
    # Count occurrences of each feature
    feature_counts = {}
    for features_with_ranks in feature_sets_with_ranks:
        # Get unique feature IDs from this set
        feature_ids = set(feature_id for feature_id, _ in features_with_ranks)
        for feature_id in feature_ids:
            if feature_id in feature_counts:
                feature_counts[feature_id] += 1
            else:
                feature_counts[feature_id] = 1
    
    # Filter features by minimum occurrences
    common_features = [feature_id for feature_id, count in feature_counts.items() 
                      if count >= min_occurrences]
    
    return common_features

def process_taxonomy(model_name: str, taxonomy: str, layers: List[int], 
                     feature_selection: str = "ranking", min_occurrences: int = 2,
                     tokenizer=None, model=None) -> Tuple[Dict[int, Dict[int, dict]], Dict[int, Dict[int, dict]], Dict[str, int]]:
    """
    Process a taxonomy by generating sentences, analyzing with GemmaScope,
    finding common features, and fetching feature data.
    
    Args:
        model_name: Name of the model
        taxonomy: The taxonomy to process
        layers: List of layer indices to analyze
        feature_selection: Method for selecting features ('ranking', 'intersection', or 'union')
        min_occurrences: Minimum number of occurrences for intersection method
        tokenizer: Pre-loaded tokenizer (optional)
        model: Pre-loaded model (optional)
        
    Returns:
        Tuple containing:
        1. Dictionary mapping layer indices to filtered feature data
        2. Dictionary mapping layer indices to unfiltered feature data
        3. Statistics dictionary
    """
    sentences = generate_taxonomy_sentences(taxonomy)
    
    # Initialize a dictionary to store feature IDs with ranks for each layer and sentence
    layer_sentence_features = {layer: [] for layer in layers}
    
    print(f"Processing taxonomy: {taxonomy}")
    for sentence, token in tqdm(sentences):
        layer_features = analyze_with_gemmascope(model_name, sentence, token, layers, tokenizer, model)
        
        for layer, feature_ids_with_ranks in layer_features.items():
            layer_sentence_features[layer].append(feature_ids_with_ranks)
    
    # Select features for each layer based on the specified method
    top_features = {}
    for layer, feature_sets_with_ranks in layer_sentence_features.items():
        print(f"\nLayer {layer}: Selecting features using method '{feature_selection}'")
        
        if feature_selection == "ranking":
            # Get weighted scores for features
            feature_scores = count_weighted_features(feature_sets_with_ranks)
            
            # Print feature scores for this layer
            print(f"Layer {layer} feature scores (top 20):")
            for feature_id, scores in sorted(feature_scores.items(), key=lambda x: x[1]['weighted_score'], reverse=True)[:20]:
                print(f"  Feature {feature_id}: count={scores['count']}, weighted_score={scores['weighted_score']:.4f}")
            
            # Get the top 16 features by weighted score (requiring at least 2 occurrences)
            top_n = min(16, len(feature_scores))
            if top_n > 0:
                layer_top_features = get_top_weighted_features(feature_scores, top_n, min_count=min_occurrences)
                print(f"Top {top_n} features for layer {layer} (by weighted score):")
                for i, feature_id in enumerate(layer_top_features):
                    scores = feature_scores[feature_id]
                    print(f"  {i+1}. Feature {feature_id}: count={scores['count']}, weighted_score={scores['weighted_score']:.4f}")
                top_features[layer] = layer_top_features
            else:
                print(f"Layer {layer}: No features found")
                
        elif feature_selection == "intersection":
            # Get features that appear in at least min_occurrences sets
            common_features = get_intersection_features(feature_sets_with_ranks, min_occurrences)
            if common_features:
                print(f"Found {len(common_features)} features in at least {min_occurrences} examples for layer {layer}")
                print(f"Common features: {common_features[:20]}" + ("..." if len(common_features) > 20 else ""))
                top_features[layer] = common_features
            else:
                print(f"Layer {layer}: No features found in at least {min_occurrences} examples")
                
        elif feature_selection == "union":
            # Get union of all features
            all_features = get_union_features(feature_sets_with_ranks)
            if all_features:
                print(f"Found {len(all_features)} unique features across all examples for layer {layer}")
                print(f"Sample features: {all_features[:20]}" + ("..." if len(all_features) > 20 else ""))
                top_features[layer] = all_features
            else:
                print(f"Layer {layer}: No features found")
        
        else:
            raise ValueError(f"Unknown feature selection method: {feature_selection}")
    
    # Fetch feature data for top features
    filtered_result = {}
    unfiltered_result = {}
    total_features_processed = 0
    total_features_aligned = 0
    
    print(f"\n{'='*80}")
    print(f"FETCHING FEATURE DATA FOR TAXONOMY: {taxonomy}")
    print(f"{'='*80}")
    
    for layer, feature_ids in top_features.items():
        print(f"\n{'-'*40}")
        print(f"Layer {layer}: Fetching data for {len(feature_ids)} features")
        print(f"{'-'*40}")
        
        layer_features_processed = 0
        layer_features_aligned = 0
        filtered_feature_data = {}
        unfiltered_feature_data = {}
        
        for feature_id in feature_ids:
            # Get URL for the feature
            url = get_neuronpedia_urls(model_name, layer, [feature_id])[feature_id]
            
            # Fetch the feature data
            data = fetch_feature(url)
            if data:
                total_features_processed += 1
                layer_features_processed += 1
                
                # Store unfiltered data (before alignment check)
                unfiltered_feature_data[feature_id] = {
                    'explanation': data['explanation'],
                    'top_activations': data['top_activations']
                }
                
                # Check if the feature aligns with the taxonomy
                if check_feature_alignment(data, taxonomy):
                    # Store filtered data (after alignment check)
                    filtered_feature_data[feature_id] = {
                        'explanation': data['explanation'],
                        'top_activations': data['top_activations']
                    }
                    
                    total_features_aligned += 1
                    layer_features_aligned += 1
                    
                    # Print detailed information about this feature
                    explanation = data['explanation']
                    # Truncate explanation if too long
                    if len(explanation) > 100:
                        explanation = explanation[:97] + "..."
                    
                    print(f"Feature ID: {feature_id} [ALIGNED WITH TAXONOMY]")
                    print(f"  Explanation: {explanation}")
                    print(f"  Top activations: {len(data['top_activations'])} examples")
                    
                    # Print the first activation example
                    if data['top_activations']:
                        first_example = data['top_activations'][0]
                        if len(first_example) > 100:
                            first_example = first_example[:97] + "..."
                        print(f"  Example activation: {first_example}")
                else:
                    print(f"Feature ID: {feature_id} [NOT ALIGNED WITH TAXONOMY - SKIPPED]")
                
                print()  # Empty line for readability
        
        if filtered_feature_data:
            filtered_result[layer] = filtered_feature_data
        
        if unfiltered_feature_data:
            unfiltered_result[layer] = unfiltered_feature_data
            
        filtered_out = layer_features_processed - layer_features_aligned
        print(f"Layer {layer} summary: {layer_features_aligned} features aligned, {filtered_out} filtered out ({layer_features_aligned}/{layer_features_processed} = {layer_features_aligned/layer_features_processed:.1%} kept)")
    
    # Print overall summary for this taxonomy
    filtered_out = total_features_processed - total_features_aligned
    keep_percentage = (total_features_aligned / total_features_processed * 100) if total_features_processed > 0 else 0
    print(f"\nTaxonomy '{taxonomy}' summary:")
    print(f"- Total features processed: {total_features_processed}")
    print(f"- Features aligned with taxonomy: {total_features_aligned}")
    print(f"- Features filtered out: {filtered_out}")
    print(f"- Percentage of features kept: {keep_percentage:.1f}%")
    
    print(f"{'='*80}\n")
    return filtered_result, unfiltered_result, {'processed': total_features_processed, 'aligned': total_features_aligned}

def save_results(model_name: str, results: Dict[str, Dict[int, Dict[int, dict]]], 
                unfiltered_results: Dict[str, Dict[int, Dict[int, dict]]] = None,
                stats: Dict[str, Dict] = None, output_dir: str = None, save_unfiltered: bool = False):
    """
    Save the results to JSON files, with one file per layer.
    
    Args:
        model_name: Name of the model
        results: Dictionary mapping taxonomies to layer-feature data (filtered)
        unfiltered_results: Dictionary mapping taxonomies to layer-feature data (unfiltered)
        stats: Dictionary containing feature filtering statistics
        output_dir: Directory to save results
        save_unfiltered: Whether to save unfiltered results
    """
    if output_dir is None:
        output_dir = "../features/sae_features"
    
    model_dir = f"{output_dir}/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create unfiltered directory if needed
    if save_unfiltered and unfiltered_results:
        unfiltered_dir = f"{output_dir}_unfiltered/{model_name}"
        os.makedirs(unfiltered_dir, exist_ok=True)
    
    # Track stats for summary
    total_files = 0
    taxonomy_counts = {}
    
    # Save filtered results layer by layer
    for taxonomy, layer_data in results.items():
        taxonomy_dir = f"{model_dir}/{taxonomy}"
        os.makedirs(taxonomy_dir, exist_ok=True)
        
        # Create unfiltered taxonomy directory if needed
        if save_unfiltered and unfiltered_results and taxonomy in unfiltered_results:
            unfiltered_taxonomy_dir = f"{unfiltered_dir}/{taxonomy}"
            os.makedirs(unfiltered_taxonomy_dir, exist_ok=True)
        
        taxonomy_counts[taxonomy] = 0
        
        # Create a simple summary for this taxonomy
        taxonomy_summary = {}
        unfiltered_taxonomy_summary = {}
        
        for layer, feature_data in layer_data.items():
            # Create a serializable version of the feature data
            serializable_data = {}
            for feature_id, data in feature_data.items():
                serializable_data[str(feature_id)] = data
            
            # Save to a layer-specific file
            output_file = f"{taxonomy_dir}/layer_{layer}_features.json"
            with open(output_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            # Add feature IDs to summary
            taxonomy_summary[str(layer)] = list(map(int, feature_data.keys()))
            
            print(f"Saved layer {layer} features for {taxonomy} to {output_file}")
            total_files += 1
            taxonomy_counts[taxonomy] = taxonomy_counts.get(taxonomy, 0) + 1
            
            # Save unfiltered data if available
            if save_unfiltered and unfiltered_results and taxonomy in unfiltered_results and layer in unfiltered_results[taxonomy]:
                unfiltered_feature_data = unfiltered_results[taxonomy][layer]
                unfiltered_serializable_data = {}
                for feature_id, data in unfiltered_feature_data.items():
                    unfiltered_serializable_data[str(feature_id)] = data
                
                unfiltered_output_file = f"{unfiltered_taxonomy_dir}/layer_{layer}_features.json"
                with open(unfiltered_output_file, 'w') as f:
                    json.dump(unfiltered_serializable_data, f, indent=2)
                
                unfiltered_taxonomy_summary[str(layer)] = list(map(int, unfiltered_feature_data.keys()))
                
                print(f"Saved unfiltered layer {layer} features for {taxonomy} to {unfiltered_output_file}")
        
        # Save the summary file for this taxonomy
        summary_file = f"{taxonomy_dir}/sae_features_{taxonomy}.json"
        with open(summary_file, 'w') as f:
            json.dump(taxonomy_summary, f, indent=2)
        
        print(f"Saved summary for {taxonomy} to {summary_file}")
        
        # Save unfiltered summary if available
        if save_unfiltered and unfiltered_results and taxonomy in unfiltered_results:
            unfiltered_summary_file = f"{unfiltered_taxonomy_dir}/sae_features_{taxonomy}.json"
            with open(unfiltered_summary_file, 'w') as f:
                json.dump(unfiltered_taxonomy_summary, f, indent=2)
            
            print(f"Saved unfiltered summary for {taxonomy} to {unfiltered_summary_file}")
    
    # Create a global summary file for filtered results
    if stats:
        summary = {
            "model": model_name,
            "taxonomies": {},
            "total_files": total_files,
            "filtering_stats": {
                "total_processed": sum(stat_data['processed'] for stat_data in stats.values()),
                "total_aligned": sum(stat_data['aligned'] for stat_data in stats.values()),
                "total_filtered": sum(stat_data['processed'] - stat_data['aligned'] for stat_data in stats.values())
            }
        }
        
        # Calculate overall percentage
        total_processed = summary["filtering_stats"]["total_processed"]
        if total_processed > 0:
            summary["filtering_stats"]["percentage_kept"] = round(summary["filtering_stats"]["total_aligned"] / total_processed * 100, 1)
        else:
            summary["filtering_stats"]["percentage_kept"] = 0
        
        for taxonomy, count in taxonomy_counts.items():
            if taxonomy in stats:
                summary["taxonomies"][taxonomy] = {
                    "layers_saved": count,
                    "features_processed": stats[taxonomy]['processed'],
                    "features_aligned": stats[taxonomy]['aligned'],
                    "features_filtered": stats[taxonomy]['processed'] - stats[taxonomy]['aligned'],
                    "percentage_kept": round(stats[taxonomy]['aligned'] / stats[taxonomy]['processed'] * 100, 1) if stats[taxonomy]['processed'] > 0 else 0
                }
        
        summary_file = f"{model_dir}/summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Create unfiltered summary too
        if save_unfiltered and unfiltered_results:
            unfiltered_summary = {
                "model": model_name,
                "taxonomies": {},
                "total_files": sum(1 for taxonomy in unfiltered_results for layer in unfiltered_results[taxonomy]),
                "note": "These are unfiltered results (before alignment checking)"
            }
            
            for taxonomy in unfiltered_results:
                layer_count = len(unfiltered_results[taxonomy])
                feature_count = sum(len(features) for features in unfiltered_results[taxonomy].values())
                
                unfiltered_summary["taxonomies"][taxonomy] = {
                    "layers_saved": layer_count,
                    "total_features": feature_count
                }
            
            unfiltered_summary_file = f"{unfiltered_dir}/summary.json"
            with open(unfiltered_summary_file, 'w') as f:
                json.dump(unfiltered_summary, f, indent=2)
        
        # Print final summary with filtering statistics
        print(f"\n{'-'*40}")
        print(f"FINAL RESULTS SUMMARY:")
        print(f"{'-'*40}")
        print(f"- Model: {model_name}")
        print(f"- Total files saved: {total_files}")
        
        print("\nFiltering statistics:")
        print(f"- Total features processed: {summary['filtering_stats']['total_processed']}")
        print(f"- Features aligned and kept: {summary['filtering_stats']['total_aligned']}")
        print(f"- Features filtered out: {summary['filtering_stats']['total_filtered']}")
        print(f"- Overall percentage of features kept: {summary['filtering_stats']['percentage_kept']}%")
        
        print("\nBy taxonomy:")
        for taxonomy, data in summary["taxonomies"].items():
            print(f"- {taxonomy}: {data['layers_saved']} layers, {data['features_aligned']}/{data['features_processed']} features kept ({data['percentage_kept']}%)")
        
        print(f"\nSummary saved to {summary_file}")
        
        if save_unfiltered and unfiltered_results:
            print(f"\nUnfiltered results saved to {unfiltered_dir}")
            print(f"Unfiltered summary saved to {unfiltered_summary_file}")

def main(model_name="gemma-2-9b", layers=None, taxonomies=None, output_dir=None, 
         feature_selection="ranking", min_occurrences=2, batch_size=1, debug=False, verbose=False):
    """
    Main function to run the entire pipeline.
    
    Args:
        model_name: Name of the model
        layers: List of layer indices to analyze
        taxonomies: List of taxonomies to process (None for all)
        output_dir: Directory to save results
        feature_selection: Method for selecting features ('ranking', 'intersection', or 'union')
        min_occurrences: Minimum number of occurrences for intersection method
        batch_size: Batch size for processing
        debug: Enable debug mode
        verbose: Enable verbose logging of feature data
    """
    # Set default layers if not provided
    if layers is None:
        layers = list(range(10, 30))  # Example: analyze layers 10-29
    
    # Get taxonomies
    all_taxonomies = get_taxonomies()
    if taxonomies is None:
        taxonomies = list(all_taxonomies.keys())
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = "../features/sae_features"
    
    # Validate feature selection method
    valid_methods = ["ranking", "intersection", "union"]
    if feature_selection not in valid_methods:
        print(f"Warning: Invalid feature selection method '{feature_selection}'. Using 'ranking' instead.")
        feature_selection = "ranking"
    
    if debug:
        print(f"Model: {model_name}")
        print(f"Layers: {layers}")
        print(f"Taxonomies: {taxonomies}")
        print(f"Feature selection method: {feature_selection}")
        print(f"Minimum occurrences: {min_occurrences}")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {batch_size}")
    
    # Load model and tokenizer once
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Determine model configuration
        if "gemma-2-9b" in model_name or "10b" in model_name:
            model_id = "google/gemma-2-9b"
        elif "gemma-2-2b" in model_name:
            model_id = "google/gemma-2-2b"
        elif "Llama-3.1-8B" in model_name:
            model_id = "meta-llama/Llama-3.1-8B-Instruct"
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        print(f"Loading tokenizer and model: {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
        
        if device == "cuda":
            print("Model loaded to GPU")
        else:
            print("Running on CPU (this will be slow)")
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        tokenizer = None
        model = None
    
    # Process each taxonomy
    filtered_results = {}
    unfiltered_results = {}
    stats = {}
    for taxonomy in taxonomies:
        if taxonomy not in all_taxonomies:
            print(f"Skipping unknown taxonomy: {taxonomy}")
            continue
        
        filtered_taxonomy_results, unfiltered_taxonomy_results, taxonomy_stats = process_taxonomy(
            model_name=model_name, 
            taxonomy=taxonomy, 
            layers=layers, 
            feature_selection=feature_selection, 
            min_occurrences=min_occurrences,
            tokenizer=tokenizer, 
            model=model
        )
        filtered_results[taxonomy] = filtered_taxonomy_results
        unfiltered_results[taxonomy] = unfiltered_taxonomy_results
        stats[taxonomy] = taxonomy_stats
    
    # Save results - both filtered and unfiltered
    save_results(
        model_name=model_name, 
        results=filtered_results, 
        unfiltered_results=unfiltered_results,
        stats=stats, 
        output_dir=output_dir,
        save_unfiltered=True  # Always save unfiltered results
    )
    
    # Free memory
    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

