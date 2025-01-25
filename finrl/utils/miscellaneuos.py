import hashlib

def stable_hash(data):
    return hashlib.sha256(str(data).encode()).hexdigest()