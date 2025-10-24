# Lead Engineer Take-Home Assessment

Thank you for your interest in joining our team.

This assessment is designed to give us a practical understanding of your engineering skills in a real-world context. It reflects a common engineering task: taking a development prototype and preparing it for production.

## The Scenario

Your team is responsible for a new microservice: the "Product Image Processor." Its purpose is to standardise and optimise product images for an e-commerce platform.

You'll find that the current prototype code has various issues ranging from bugs to missing implementations. The codebase has amassed technical debt over multiple iterations, rotating developers, and bouts of vibe coding. Part of your task is identifying and addressing these issues whilst maintaining the intended functionality. The prototype is now moving towards production, and your task is to take ownership of the prototype codebase. You will need to **audit the existing codebase, identify and fix bugs and anti-patterns, and implement enhancements to make the service robust, secure, and scalable.**

### Tech Stack

This assessment uses:

- FastAPI for the web framework
- LocalStack to simulate AWS DynamoDB locally
- Terraform for infrastructure as code
- Docker and docker-compose for containerisation

While we prefer you use the technologies listed above, if you have a compelling reason to use an alternative (such as Flask instead of FastAPI or CloudFormation instead of Terraform), you may do so. If you do use a different stack, please include an explanation of your choices in your submission.

## Assessment Guidelines

### Estimated time & partial completion

We estimate candidates should spend around **6 hours** on this task. This is a guideline, we are most interested in the quality of your analysis and the reasoning behind your fixes. Quality is important, but we also value practical solutions over perfect ones. We understand that engineering tasks can be unpredictable. If you find yourself spending too much time on a particular component, we encourage you to submit what you have - we value seeing your approach and thought process even in partial solutions.

### Starter code

**The provided starter code is intentionally flawed and incomplete.** This is a deliberate part of the assessment designed to simulate taking over a prototype codebase that needs production-readiness work. You are encouraged to:

- Modify any existing code as needed
- Use your engineering judgement to make practical decisions about implementation details
- Add any dependencies or configuration changes required for a working solution

**The goal is not to work around the existing code's limitations**, but rather to take ownership of it and improve it to production standards. If something in the starter code doesn't make sense or seems broken, it probably is. Fixing it is part of the exercise.

### Use of AI

We recognise that AI code assistants are becoming an integral part of modern development workflows, and their use is therefore permitted during this assessment. However, your ability to understand, debug, and explain the submitted code remains crucial.

### Development environment

The project includes LocalStack configuration to simulate AWS services locally, allowing you to develop and test **without requiring actual AWS resources**. We do not expect candidates to deploy to AWS or incur any costs as part of this exercise. LocalStack will allow you to:

- Provision and interact with a DynamoDB table without AWS credentials.
- Develop the complete solution on your local machine.
- Ensure your Terraform configurations work correctly before any real AWS deployment.

If you are attempting the assessment on your own device, you will need to install the following dependencies: `Docker` (`Podman` accepted), `Docker Compose`, `terraform`, `LocalStack` and `Python`.

### Deliverables

Your submission should include:

1.  All modified source code files.
2.  Updated infrastructure configuration.
3.  A brief README or documentation file explaining:
    - Key issues identified and how you addressed them.
    - Any design decisions you made and why.
    - Any TODOs or recommendations for future improvements.
    - Instructions for running your solution.

---

## Assessment Part 1: Application & API Refinement

A service has been developed to support the primary user interface. This UI offers two ways to process product photos:

1.  **Manual Crop:** A content manager can upload a photo and use a visual tool to draw a precise crop box. When they save their selection, the UI calls the `manual-crop` endpoint.

2.  **Smart Crop:** For faster processing, the manager can simply upload a photo and click a "Smart Crop" button. This action, which doesn't require manual selection, calls the `smart-crop` endpoint. This endpoint relies on a mock "AI service" and is expected to receive high traffic, so it must be protected against abuse.

### Functional Requirements:

**Storage:**  
The application must save all processed images to a persistent location (`/app/processed_images` within the container). In a real production deployment, this path would be a volume mounted to a shared network filesystem (like NFS or AWS EFS). Therefore, you do not need to implement object storage (like S3); focus on making the file I/O within the service robust. You are still welcome to suggest other architectural improvements in a TODO documentation section.

**Manual Crop Endpoint (`POST /images/manual-crop`):**

- **Request:** `multipart/form-data` containing:
  1.  A file field named `source_image`
  2.  A JSON string field `product_info` (e.g., `{"product_id": "ABC-123"}`)
  3.  A JSON string field `crop_box` (e.g., `{"x": 10, "y": 10, "width": 200, "height": 200}`)
- **Behaviour:** Must synchronously perform the crop, save the resulting image, and return a JSON response.
- **Success Response (`200 OK`):** Must be JSON with the following structure. The `retrieval_url` must be structured so it is resolvable by the `GET /images/{image_path:path}` endpoint.
  ```json
  {
    "image_id": "your-unique-image-id",
    "retrieval_url": "/images/<path_to_your_image>"
  }
  ```

**Smart Crop Endpoint (`POST /images/smart-crop`):**

- **Request:** `multipart/form-data` containing:
  1.  A file field named `source_image`
  2.  A JSON string field `product_info` (e.g., `{"product_id": "ABC-123"}`)
- **Behaviour:**
  - This is a high-traffic endpoint that needs protection. You must implement a **rate-limiting** mechanism before any processing occurs.
  - The rate limit is **20 requests per hour for each source IP address**. The state must be stored in a **DynamoDB table**.
  - If the rate limit is exceeded, the service must return an error immediately.
  - If the request is within the limit, it must **immediately return an ID for the eventual result and process the image in the background.**
- **Background Processing:** The processing logic (which calls the mock AI service) must be decoupled to run as a background task.
- **Rate Limit Exceeded Response (`429 Too Many Requests`):** Must be a JSON response with a clear error message.
- **Success Response (`202 Accepted`):** Must be JSON with the following structure.
  ```json
  {
    "image_id": "the-unique-id-for-the-future-image",
    "retrieval_url": "/images/<path_to_your_image>"
  }
  ```
  - Note that because processing happens in the background, a GET request to the `retrieval_url` is expected to return a `404 Not Found` until the background task is complete. The frontend client (out of scope for this assessment) is responsible for polling this URL.

**Image Retrieval Endpoint (`GET /images/{image_path:path}`):**

- **Behaviour:**
  - This endpoint retrieves an image using a flexible path. The `image_path` parameter represents the **full relative path** of the file within the `/app/processed_images` directory.
  - On success: Return `200 OK` with `Content-Type: image/jpeg` and the raw binary image data.
  - If the path does not correspond to an existing image: Return `404 Not Found`.

---

## Assessment Part 2: Containerisation & Infrastructure as Code

The current container and infrastructure definitions are partly functional but are not optimised for production.

**Containerisation (`Dockerfile`, `docker-compose.yml`):**

- Ensure that processed images stored in `/app/processed_images` persist across container restarts during local development by appropriately configuring the `docker-compose.yml` file.
- Beyond this, for both files:
  - Review and improve these files following containerisation best practices that you are aware of.
  - Consider security, efficiency, developer experience, and maintainability in your improvements.

**Infrastructure as Code (`*.tf` files):**

- Complete the Terraform configuration to deploy the **DynamoDB table** required for the rate-limiting feature to LocalStack.
- The table must have a schema suitable for storing rate-limiting data per IP address. We suggest a table with a **string partition key named `ip_address`**.
- Ensure the configuration works with LocalStack for local development.

---

## Assessment Part 3: Automation

**Automation (`Makefile`):**

- Review and ensure the `Makefile` targets are reliable and correctly report success or failure. The current Makefile is flawed and is negatively impacting the developer experience. The team plans to move to GitHub Actions (no need to implement this) in the future but for now the current workflow can benefit from an improved makefile.
- The Makefile should provide targets that, at minimum:
  - Run the application locally.
  - Deploy the infrastructure (the DynamoDB table) to LocalStack.
  - Build the application container (pushing to a container repository is not required).

---

Good Luck!
