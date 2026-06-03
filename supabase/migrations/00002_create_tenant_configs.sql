CREATE TABLE tenant_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    config JSONB NOT NULL,
    config_version TEXT NOT NULL,
    validated_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_tenant_configs_tenant_id ON tenant_configs(tenant_id);

ALTER TABLE tenant_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to tenant_configs"
    ON tenant_configs
    FOR ALL
    USING (auth.role() = 'service_role');
