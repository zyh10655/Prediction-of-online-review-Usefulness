-- foreign keys not enabled by default in sqlite
PRAGMA foreign_keys = ON;

CREATE TABLE business (
    business_id TEXT PRIMARY KEY,
    name TEXT,
    address TEXT,
    city TEXT,
    state TEXT,
    postal_code TEXT,
    latitude NUMERIC, -- numeric type only first 15 s.f. kept
    longitude NUMERIC,
    stars NUMERIC,
    review_count INTEGER,
    is_open INTEGER,
    attributes TEXT,
    categories TEXT,
    hours TEXT
);

CREATE TABLE user (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    review_count INTEGER,
    yelping_since TEXT,
    friends_count INTEGER,
    useful INTEGER,
    funny INTEGER,
    cool INTEGER,
    fans INTEGER,
    elite TEXT,
    average_stars NUMERIC,
    compliment_hot INTEGER,
    compliment_more INTEGER,
    compliment_profile INTEGER,
    compliment_cute INTEGER,
    compliment_list INTEGER,
    compliment_note INTEGER,
    compliment_plain INTEGER,
    compliment_cool INTEGER,
    compliment_funny INTEGER,
    compliment_writer INTEGER,
    compliment_photos INTEGER
);

CREATE TABLE review (
    review_id TEXT PRIMARY KEY,
    user_id TEXT,
    business_id TEXT,
    stars INTEGER,
    date TEXT,
    text TEXT,
    useful INTEGER,
    funny INTEGER,
    cool INTEGER,
    FOREIGN KEY (user_id) REFERENCES user,
    FOREIGN KEY (business_id) REFERENCES business
);
