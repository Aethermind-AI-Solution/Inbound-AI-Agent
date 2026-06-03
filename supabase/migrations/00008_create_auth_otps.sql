CREATE TABLE auth_otps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    caller_id UUID NOT NULL REFERENCES callers(id) ON DELETE CASCADE,
    code_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    attempts INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_auth_otps_caller ON auth_otps(caller_id);
CREATE INDEX idx_auth_otps_expires ON auth_otps(expires_at);

ALTER TABLE auth_otps ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to auth_otps"
    ON auth_otps
    FOR ALL
    USING (auth.role() = 'service_role');
