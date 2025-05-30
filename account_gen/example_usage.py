from reddit_bot import RedditBot
import random
import string


def generate_random_string(length=10):
    """Generate a random string for usernames and passwords"""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def main():
    # Initialize the bot with your Reddit clone's base URL
    base_url = "http://your-reddit-clone.com"  # Replace with your actual URL
    bot = RedditBot(base_url)

    try:
        # Generate random credentials
        username = f"test_user_{generate_random_string(8)}"
        email = f"{username}@example.com"
        password = generate_random_string(12)

        # Create a new account
        print(f"Creating account for {username}...")
        if bot.create_account(username, email, password):
            print("Account created successfully!")

            # Create a post
            subreddit = "test"
            title = "Test Post Title"
            content = "This is a test post content."
            print(f"Creating post in r/{subreddit}...")
            if bot.create_post(subreddit, title, content):
                print("Post created successfully!")

                # Add a comment
                comment = "This is a test comment."
                print("Adding comment...")
                if bot.comment_on_post(
                    f"{base_url}/r/{subreddit}/comments/your-post-id", comment
                ):
                    print("Comment added successfully!")

                # Vote on the post
                print("Voting on post...")
                if bot.vote_on_post(
                    f"{base_url}/r/{subreddit}/comments/your-post-id", "up"
                ):
                    print("Vote recorded successfully!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        bot.close()


if __name__ == "__main__":
    main()
