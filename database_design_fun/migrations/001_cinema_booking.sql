CREATE TABLE screens (
    id SERIAL PRIMARY KEY,
    screen_id INTEGER NOT NULL, -- screens generally tend to have a number id
    theatre_id INTEGER NOT NULL,
    FOREIGN KEY (theatre_id) REFERENCES theatres(id) ON DELETE CASCADE,
    -- other information like screen size etc
);

CREATE TABLE theatres (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    -- other information like address etc
);

CREATE TABLE seats (
    id SERIAL PRIMARY KEY,
    screen_id INTEGER NOT NULL,
    seat_id INTEGER NOT NULL,
    theatre_id INTEGER NOT NULL,
    FOREIGN KEY (screen_id) REFERENCES screens(id) ON DELETE CASCADE,
    FOREIGN KEY (theatre_id) REFERENCES theatres(id), -- don't need to cascade delete as handled by screen
    -- other information like seat type etc
);

CREATE TABLE movies (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    -- other movie information
);

CREATE TABLE screenings (
    id SERIAL PRIMARY KEY,
    movie_id INTEGER NOT NULL,
    screen_id INTEGER NOT NULL,
    theatre_id INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    FOREIGN KEY (movie_id) REFERENCES movies(id),
    FOREIGN KEY (screen_id) REFERENCES screens(id) ON DELETE CASCADE,
    FOREIGN KEY (theatre_id) REFERENCES theatres(id),
);

CREATE TYPE booking_status AS ENUM ('confirmed', 'cancelled', 'completed');

CREATE TABLE bookings (
    id SERIAL PRIMARY KEY,
    screening INTEGER NOT NULL,
    seat INTEGER NOT NULL,
    status booking_status NOT NULL DEFAULT 'confirmed',
    FOREIGN KEY (screening) REFERENCES screenings(id) ON DELETE CASCADE,
    FOREIGN KEY (seat) REFERENCES seats(id)
);

SELECT id FROM screenings
WHERE movie_id = blah blah
AND theatre_id = blah blah
as showings

SELECT seat from bookings 
WHERE id in showings
as booked_seats

SELECT id from seats
WHERE movie_id = blah blah
AND theatre_id = blah blah
AND id NOT in booked_seats

-- Find available seats for a specific screening
SELECT s.* 
FROM seats s
WHERE s.screen_id = (
    SELECT screen_id 
    FROM screenings 
    WHERE id = [screening_id]
)
AND s.id NOT IN (
    SELECT seat 
    FROM bookings 
    WHERE screening = [screening_id]
    AND status = 'confirmed'
);

SELECT s.*
FROM seats s
JOIN screenings sc on s.screen_id = sc.screen_id
WHERE sc.movie_id = 
AND sc.theatreId = 
AND s.id NOT IN
(
    SELECT seat
    FROM bookings b 
    JOIN screenings sc2 on b.screen_id = sc2.screen_id
    WHERE sc2.movie_id = blah blah
    AND sc2.theature_id = blahblah
    and b.status = 'confirmed'
)

-- SELECT s.* 
-- FROM seats s
-- JOIN screenings sc on s.screen_id = sc.screen_id --inner join
-- -- JOIN screenings sc ON s.screen_id = sc.screen_id
-- WHERE sc.movie_id = [movie_id]
-- AND sc.theatre_id = [theatre_id]
-- AND s.id NOT IN (
--     SELECT seat 
--     FROM bookings b
--     JOIN screenings sc2 ON b.screening = sc2.id
--     WHERE sc2.movie_id = [movie_id]
--     AND sc2.theatre_id = [theatre_id]
--     AND b.status = 'confirmed'
-- );
