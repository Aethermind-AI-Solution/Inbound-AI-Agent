CREATE TABLE bookings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    service_id TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    caller_id UUID NOT NULL REFERENCES callers(id),
    start_ts TIMESTAMPTZ NOT NULL,
    end_ts TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'confirmed' CHECK (status IN ('confirmed', 'cancelled', 'completed', 'no_show')),
    custom_values JSONB NOT NULL DEFAULT '{}'::JSONB,
    idempotency_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_bookings_tenant_id ON bookings(tenant_id);
CREATE INDEX idx_bookings_caller_id ON bookings(caller_id);
CREATE INDEX idx_bookings_resource_time ON bookings(resource_id, start_ts, end_ts);

ALTER TABLE bookings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role has full access to bookings"
    ON bookings
    FOR ALL
    USING (auth.role() = 'service_role');
