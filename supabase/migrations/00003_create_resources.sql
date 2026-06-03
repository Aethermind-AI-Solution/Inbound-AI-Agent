CREATE TABLE resources (
    id TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    capacity INT NOT NULL DEFAULT 1 CHECK (capacity >= 1),
    tags JSONB NOT NULL DEFAULT '[]'::JSONB,
    calendar_ref TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, tenant_id)
);

CREATE INDEX idx_resources_tenant_id ON resources(tenant_id);
CREATE INDEX idx_resources_type ON resources(tenant_id, type);

ALTER TABLE resources ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to resources"
    ON resources
    FOR ALL
    USING (auth.role() = 'service_role');
