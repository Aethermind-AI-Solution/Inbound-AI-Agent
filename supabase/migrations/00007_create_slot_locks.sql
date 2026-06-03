CREATE TABLE slot_locks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resource_id TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    start_ts TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    caller_id UUID REFERENCES callers(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (resource_id, start_ts)
);

CREATE INDEX idx_slot_locks_expires ON slot_locks(expires_at);

ALTER TABLE slot_locks ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to slot_locks"
    ON slot_locks
    FOR ALL
    USING (auth.role() = 'service_role');
