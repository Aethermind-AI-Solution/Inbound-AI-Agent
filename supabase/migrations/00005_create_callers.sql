CREATE TABLE callers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    phone TEXT NOT NULL,
    verified_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, phone)
);

CREATE INDEX idx_callers_phone ON callers(tenant_id, phone);

ALTER TABLE callers ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to callers"
    ON callers
    FOR ALL
    USING (auth.role() = 'service_role');
