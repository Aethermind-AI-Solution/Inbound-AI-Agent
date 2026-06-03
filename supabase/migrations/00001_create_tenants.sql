CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sector TEXT NOT NULL CHECK (sector IN ('salon', 'clinic', 'restaurant', 'gym', 'home_service', 'real_estate')),
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'staging', 'live')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE tenants ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to tenants"
    ON tenants
    FOR ALL
    USING (auth.role() = 'service_role');
