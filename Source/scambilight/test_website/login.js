document.getElementById('loginForm').addEventListener('submit', function(e) {
    e.preventDefault(); // Prevent the default form submission

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Construct the request payload
    const payload = {
        username: username,
        password: password
    };

    // Define the API endpoint
    const apiUrl = 'https://your-backend-api.com/login';

    // Make the API call
    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && data.token) {
            // Store the token in localStorage or sessionStorage
            sessionStorage.setItem('sessionToken', data.token);

            // Handle success - e.g., redirect to another page or display a success message
            console.log('Login successful:', data);
            window.location.href = 'your-protected-page.html'; // Redirect the user to the protected page
            // Redirect to a different page or update the UI
        } else {
            // Handle failure - e.g., display an error message
            console.error('Login failed:', data.message);
        }
    })
    .catch(error => {
        // Handle network errors or other errors that occurred during the fetch
        console.error('Error:', error);
    });
});
