import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Hardcoded device IDs from device-tag-mapping.json
const deviceIds = ['abc12345', 'bc1ad358bb', '54e5b53659'];

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

export let options = {
  stages: [
    // Warm up
    { duration: '30s', target: 100 },
    
    // Gradual ramp up to find breaking point
    // { duration: '5s', target: 25 },
    // { duration: '5s', target: 100 },
    // { duration: '5s', target: 200 },
    // { duration: '5s', target: 300 },
    // { duration: '5s', target: 500 },
    // { duration: '1m', target: 500 },   // Hold at 500 users to see if it stabilizes
    
    // // Push it to the limit
    // { duration: '30s', target: 750 },
    // { duration: '30s', target: 1000 },
    // { duration: '1m', target: 1000 },  // Hold at 1000 users - this will likely break it
    
    // // Ramp down
    // { duration: '30s', target: 0 },
  ],
  
  thresholds: {
    // Define what we consider "broken"
    http_req_duration: ['p(95)<1000'], // 95% of requests should be under 1 second
    http_req_failed: ['rate<0.1'],     // Error rate should be under 10%
    errors: ['rate<0.1'],              // Custom error rate under 10%
    response_time: ['p(95)<1000'],     // 95% of responses under 1 second
  },
  
  // Show detailed summary with RPS and response times
  summaryTrendStats: ['avg', 'min', 'med', 'max', 'p(90)', 'p(95)', 'p(99)'],
  
  // Keep response bodies for validation
  discardResponseBodies: false,
  
  // Ramp up slowly to find the breaking point
  noConnectionReuse: false,
  userAgent: 'k6-gamestate-destruction-test/1.0',
};

const BASE_URL = 'http://LIELLOMEN:8080';
const GAMESTATE_ENDPOINT = `${BASE_URL}/api/v1/gamestate`;

export default function() {
  const startTime = Date.now();
  
  // Make request to gamestate endpoint
  const response = http.get(GAMESTATE_ENDPOINT, {
    headers: {
      'Accept': 'application/json',
      'User-Agent': 'k6-destruction-test',
      'x-device-ID': deviceIds[Math.floor(Math.random() * deviceIds.length)], // Random device from mapping
    },
    timeout: '10s', // 10 second timeout
  });
  
  const endTime = Date.now();
  const duration = endTime - startTime;
  
  // Record custom metrics
  errorRate.add(response.status !== 200);
  responseTime.add(duration);
  
  // Check response
  const isSuccess = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
    'response time < 1000ms': (r) => r.timings.duration < 1000,
    'response time < 2000ms': (r) => r.timings.duration < 2000,
    'response has body': (r) => r.body && r.body.length > 0,
    'response is JSON': (r) => {
      if (!r.body || r.body.length === 0) return false;
      try {
        JSON.parse(r.body);
        return true;
      } catch (e) {
        return false;
      }
    },
  });
  
  // Log failures for debugging
  if (!isSuccess) {
    const bodyText = response.body ? response.body.substring(0, 200) : 'No response body';
    console.log(`FAILED REQUEST:
    Status: ${response.status}
    Duration: ${response.timings.duration}ms
    Body: ${bodyText}...
    Timestamp: ${new Date().toISOString()}`);
  }
  
  // Small sleep to avoid overwhelming the server too quickly
  sleep(0.1);
}

// Setup function - runs once at the beginning
export function setup() {
  console.log('🚀 Starting gamestate destruction test...');
  console.log('📊 This test will gradually increase load to find the breaking point');
  console.log('🎯 Target endpoint:', GAMESTATE_ENDPOINT);
  
  // Test if server is running
  const testResponse = http.get(GAMESTATE_ENDPOINT, {
    headers: {
      'x-device-ID': deviceIds[0] // Use first device for setup test
    }
  });
  if (testResponse.status !== 200) {
    throw new Error(`Server not responding! Status: ${testResponse.status}`);
  }
  
  console.log('✅ Server is responding, starting destruction test...');
  return { startTime: Date.now() };
}

// Teardown function - runs once at the end
export function teardown(data) {
  const totalTime = (Date.now() - data.startTime) / 1000;
  console.log(`🏁 Destruction test completed in ${totalTime} seconds`);
  console.log('📈 Check the summary below to see where your server broke!');
}
