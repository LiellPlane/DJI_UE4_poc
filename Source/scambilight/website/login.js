let targetUrl   = 'https://api.scambilight.com/hello';

document.getElementById("lname3").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        submitForm();
    }
});

document.getElementById("lname2").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        var inputField = document.getElementById("lname3");
        inputField.focus();
    }
});

function submitForm() {
    // Get form data
    
    
    var form = document.getElementById("passwordform");
    var formData = new FormData(form);
    let valuesValid = true;

    formData.forEach((value) => {
        if (value === null || (typeof value === "string" && value.trim() === "")) {
            valuesValid = false;
        }
    });

    if (!valuesValid) {
        alert("One or more values in the FormData are null or empty.");
    }
    // Convert formData to JSON
    var jsonData = {};
    formData.forEach((value, key) => {
        jsonData[key] = value;
    });

    // Nest the JSON object under "login" key
    let requestData = { "login": jsonData };

        fetch(targetUrl, {
            method: "POST",
            headers: {'Content-Type': 'application/json'}, 
            body: JSON.stringify(requestData)
        }).then(response => response.json())
            .then(json => {console.log(JSON.stringify(json));
            //alert(JSON.stringify(json));
            // Check if "sessionbody" key exists in the JSON response
            if (json.hasOwnProperty("sessiontoken")) {
                // Store the value of "sessionbody" in sessionStorage
                sessionStorage.setItem("sessiontoken", JSON.stringify(json.sessiontoken));
                window.location.href = 'index.html';
            }
            else
            {
                alert(JSON.stringify(json));
            }
            });
    }
    window.onload = () => {
        test_logged_in();
    };
    
    function test_logged_in() {
        console.log("testing if logged in");
        let data = {
            "action": "check_logged_in",
        };
        data = addSessionTokenIfExists(data);
    
        fetch(targetUrl, {
            method: "POST",
            headers: {'Content-Type': 'application/json'}, 
            body: JSON.stringify(data)
        })
        .then(response => {
            if (response.ok) {
                // If the status code is 200, redirect to a new page
                window.location.href = 'index.html';
            } else {
                return response.json();
            }
        })
        .then(json => {
            // Handle the JSON response as needed
            console.log(json);
        })
        .catch(error => {
            // Handle any fetch errors
            console.error('Fetch error:', error);
        });
    }

function addSessionTokenIfExists(inputDictionary) {
    const sessionToken = sessionStorage.getItem("sessiontoken");

    if (sessionToken !== null) {
        inputDictionary["sessiontoken"] = sessionToken;
        console.log("session token added to request body")
    }
    
    return inputDictionary;
}
