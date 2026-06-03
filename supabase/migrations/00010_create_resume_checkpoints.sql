CREATE TABLE resume_checkpoints (
    phone TEXT PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    state JSONB NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_resume_checkpoints_expires ON resume_checkpoints(expires_at);

ALTER TABLE resume_checkpoints ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to resume_checkpoints"
    ON resume_checkpoints
    FOR ALL
    USING (auth.role() = 'service_role');
