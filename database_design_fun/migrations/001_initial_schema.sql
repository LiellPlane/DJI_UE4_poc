CREATE TABLE brands (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    address TEXT -- example of extra info we could add
);

CREATE TABLE stores (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    location VARCHAR(255) NOT NULL,
    brand_id INTEGER NOT NULL,
    -- don't leave orphaned associations on delete of brand
    FOREIGN KEY (brand_id) REFERENCES brands(id) ON DELETE CASCADE
);

-- Products are not unique to brands
CREATE TABLE products (
    sku VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description JSONB -- can put in semi-structured data
    category VARCHAR(100) NOT NULL -- can add lots more like colour size etc
);

-- Which products are available in which stores, with store-specific pricing and stock
CREATE TABLE store_products (
    store_id INTEGER NOT NULL,
    product_sku VARCHAR(255) NOT NULL,
    cost DECIMAL(10,2) NOT NULL,
    stock INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (store_id, product_sku),
    -- don't leave orphaned associations on delete of brand/store
    FOREIGN KEY (store_id) REFERENCES stores(id) ON DELETE CASCADE,
    -- do not allow product being deleted if it is associatd with a store
    FOREIGN KEY (product_sku) REFERENCES products(sku) ON DELETE RESTRICT
);

-- read heavy, so can add indexes and not worry about write performance
CREATE INDEX ON stores(brand_id);
CREATE INDEX ON store_products(store_id);
CREATE INDEX ON store_products(product_sku);
CREATE INDEX ON store_products(store_id, product_sku);
CREATE INDEX ON products(category);
