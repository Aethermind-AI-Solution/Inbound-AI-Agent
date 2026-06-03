CREATE TABLE services (
    id TEXT NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    duration_min INT NOT NULL CHECK (duration_min > 0),
    booking_mode TEXT NOT NULL CHECK (booking_mode IN ('exclusive', 'shared')),
    price NUMERIC,
    deposit NUMERIC,
    custom_fields JSONB NOT NULL DEFAULT '[]'::JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (id, tenant_id)
);

CREATE INDEX idx_services_tenant_id ON services(tenant_id);

ALTER TABLE services ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to services"
    ON services
    FOR ALL
    USING (auth.role() = 'service_role');
