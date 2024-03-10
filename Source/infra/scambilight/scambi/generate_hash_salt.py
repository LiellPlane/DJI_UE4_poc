from typing import Tuple
import os
import hashlib
import hmac

def hash_new_password(password: str) -> Tuple[bytes, bytes]:
    """
    Hash the provided password with a randomly-generated salt and return the
    salt and hash to store in the database.
    """
    salt = os.urandom(16)
    pw_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt, pw_hash

def is_correct_password(salt: bytes, pw_hash: bytes, password: str) -> bool:
    """
    Given a previously-stored salt and hash, and a password provided by a user
    trying to log in, check whether the password is correct.
    """
    return hmac.compare_digest(
        pw_hash,
        hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    )

# # Example usage:
salt, pw_hash = hash_new_password('secretshh')

str_salt = salt.hex()

str_pwhash = pw_hash.hex()

plop=1
# salt_bytes_again = bytes.fromhex(str_salt)

# hash_pw_bytes_again = bytes.fromhex(str_pwhash)

# assert salt_bytes_again == salt
# assert hash_pw_bytes_again == pw_hash
# assert is_correct_password(salt_bytes_again, hash_pw_bytes_again, 'farts')

# assert is_correct_password(
#     bytes.fromhex("0eb8da33cb4bdb213c8bfb158ec76972"),
#     bytes.fromhex("db2602c4c3f5e160bfa09611e5f12f0172c6f1b5068e77f68bb8503434e326fc"),
#     'farts')


# assert not is_correct_password(salt, pw_hash, 'Tr0ub4dor&3')
# assert not is_correct_password(salt, pw_hash, 'rosebud')