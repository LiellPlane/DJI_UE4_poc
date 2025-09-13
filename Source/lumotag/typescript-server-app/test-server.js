const http = require('http');

// Test the server endpoints
async function testServer() {
    console.log('🧪 Testing TypeScript Game Server...\n');
    
    const baseUrl = 'http://localhost:8080';
    
    // Test 1: Root endpoint
    console.log('1️⃣ Testing root endpoint...');
    try {
        const response = await makeRequest(`${baseUrl}/`);
        console.log('✅ Root endpoint working:', JSON.stringify(response, null, 2));
    } catch (error) {
        console.log('❌ Root endpoint failed:', error.message);
        return;
    }
    
    // Test 2: Gamestate endpoint
    console.log('\n2️⃣ Testing gamestate endpoint...');
    try {
        const response = await makeRequest(`${baseUrl}/api/v1/gamestate`);
        console.log('✅ Gamestate working:', JSON.stringify(response, null, 2));
    } catch (error) {
        console.log('❌ Gamestate failed:', error.message);
    }
    
    // Test 3: Stats endpoint
    console.log('\n3️⃣ Testing stats endpoint...');
    try {
        const response = await makeRequest(`${baseUrl}/api/v1/stats`);
        console.log('✅ Stats working:', JSON.stringify(response, null, 2));
    } catch (error) {
        console.log('❌ Stats failed:', error.message);
    }
    
    // Test 4: Image upload endpoint (POST)
    console.log('\n4️⃣ Testing image upload endpoint...');
    try {
        // Create a simple test image (1x1 pixel JPEG)
        const testImageBase64 = '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwA/8A';
        
        const uploadData = {
            image_id: `test_image_${Date.now()}`,
            image_data: testImageBase64,
            event_type: 'UploadRequest',
            timestamp: Date.now()
        };
        
        const response = await makePostRequest(`${baseUrl}/api/v1/images/upload`, uploadData, {
            'X-User-ID': 'test_user',
            'Content-Type': 'application/json'
        });
        console.log('✅ Image upload working:', JSON.stringify(response, null, 2));
        
        if (response.filename) {
            console.log(`   📁 Image saved as: ${response.filename}`);
            console.log(`   📏 File size: ${response.size_bytes} bytes`);
        }
    } catch (error) {
        console.log('❌ Image upload failed:', error.message);
    }
    
    // Test 5: Events endpoint (POST)
    console.log('\n5️⃣ Testing events endpoint...');
    try {
        const eventData = {
            event_type: 'PlayerTagged',
            tag_id: 'player_001',
            image_ids: ['test_image_001']
        };
        
        const response = await makePostRequest(`${baseUrl}/api/v1/events`, eventData, {
            'X-User-ID': 'test_user',
            'Content-Type': 'application/json'
        });
        console.log('✅ Events working:', JSON.stringify(response, null, 2));
    } catch (error) {
        console.log('❌ Events failed:', error.message);
    }
    
    console.log('\n🎉 Server testing complete!');
}

function makeRequest(url) {
    return new Promise((resolve, reject) => {
        const req = http.get(url, (res) => {
            let data = '';
            res.on('data', (chunk) => data += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(data));
                } catch (error) {
                    resolve(data);
                }
            });
        });
        
        req.on('error', reject);
        req.setTimeout(5000, () => reject(new Error('Request timeout')));
    });
}

function makePostRequest(url, data, headers = {}) {
    return new Promise((resolve, reject) => {
        const postData = JSON.stringify(data);
        const urlObj = new URL(url);
        
        const options = {
            hostname: urlObj.hostname,
            port: urlObj.port,
            path: urlObj.pathname,
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData),
                ...headers
            }
        };
        
        const req = http.request(options, (res) => {
            let responseData = '';
            res.on('data', (chunk) => responseData += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(responseData));
                } catch (error) {
                    resolve(responseData);
                }
            });
        });
        
        req.on('error', reject);
        req.setTimeout(5000, () => reject(new Error('Request timeout')));
        req.write(postData);
        req.end();
    });
}

// Run the tests
testServer().catch(console.error);
