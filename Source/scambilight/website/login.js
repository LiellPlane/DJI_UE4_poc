function submitForm() {
    // Get form data
    const targetUrl   = 'https://api.scambilight.com/hello';
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
