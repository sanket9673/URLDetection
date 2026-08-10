import numpy as np
import tldextract

WHITELIST = {
    'google.com', 'github.com', 'microsoft.com', 'apple.com', 
    'youtube.com', 'wikipedia.org', 'amazon.com', 'cloudflare.com'
}

HIGH_RISK_KEYWORDS = {
    'paypal', 'verification', 'verify', 'secure', 'login', 'bank', 
    'update', 'account', 'signin', 'webscr', 'confirm', 'suspicious'
}

MALWARE_INDICATORS = {
    'payload', '.exe', '.zip', '.rar', '.tar', '.gz'
}

def compute_entropy(s):
    if not s: return 0.0
    _, counts = np.unique(list(s), return_counts=True)
    probs = counts / len(s)
    return -np.sum(probs * np.log2(probs))

def check_whitelist(url):
    """
    Checks if a URL belongs to the enterprise whitelist.
    Returns: (is_whitelisted, probability_array)
    """
    ext = tldextract.extract(url)
    # Handle subdomain edge case by extracting registered domain
    registered_domain = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
    if registered_domain in WHITELIST:
        # [Benign, Defacement, Phishing, Malware]
        return True, np.array([0.999, 0.0, 0.001, 0.0])
    return False, None

def apply_prediction_guard(url, p_final, domain_mapping):
    """
    Applies rule-based overrides for unseen domains containing suspicious keywords
    or signature malware paths to prevent false negatives.
    """
    ext = tldextract.extract(url)
    domain = f"{ext.domain}.{ext.suffix}" if ext.domain else ext.suffix
    
    # Check if domain is unseen in the original graph training data
    is_unseen = (domain not in domain_mapping)
    
    url_lower = url.lower()
    keywords_found = sum(1 for kw in HIGH_RISK_KEYWORDS if kw in url_lower)
    malware_found = sum(1 for mi in MALWARE_INDICATORS if mi in url_lower)
    entropy_val = compute_entropy(url)
    
    if is_unseen:
        # DGA/Malware payload signature check
        if malware_found >= 1 or '.exe' in url_lower or 'payload' in url_lower:
            return np.array([0.05, 0.05, 0.10, 0.80])
        # High-risk keywords or combination of entropy + keyword
        elif keywords_found >= 2 or (entropy_val > 4.2 and keywords_found >= 1):
            return np.array([0.05, 0.05, 0.80, 0.10])
            
    return p_final
