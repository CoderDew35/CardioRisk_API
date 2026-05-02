-- PostgreSQL initialisation script
-- Runs automatically when the postgres container starts for the first time

-- Ensure the database exists (already created by env vars, this is a safety net)
-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant all privileges to app user
GRANT ALL PRIVILEGES ON DATABASE cardiorisk TO cardiorisk_user;
