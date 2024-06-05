import os
import string
import secrets
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def generate_random_key(length=32):
    # Generate a random key with specified length
    alphabet = string.ascii_letters + string.digits + "!@$%^&*_+=-"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def update_secret_key():
    # Read existing SECRET_KEY from .env file
    secret_key = os.getenv('SECRET_KEY')

    # Generate a random key with length 32 (you can adjust the length as needed)
    random_key = generate_random_key(length=32)

    # Update SECRET_KEY with the new random key
    with open('.env', 'r') as f:
        lines = f.readlines()
    with open('.env', 'w') as f:
        for line in lines:
            if line.startswith('SECRET_KEY='):
                f.write(f'SECRET_KEY={random_key}\n')
            else:
                f.write(line)

    print(f'Secret key updated: {random_key}')

if __name__ == '__main__':
    update_secret_key()
