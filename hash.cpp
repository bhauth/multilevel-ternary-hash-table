
#include <iostream>

#define KEY_BYTES 8

struct TableInfo
	{
	uint32_t T2_inserts_til_resize;
	uint16_t zero_key_location;  // (1 + T2 offset) of zero key in T2; 0 if no zero key in T2
	uint16_t element_size_bytes;
	uintptr_t T1_ptr;  // element_size_bytes from start of array
	uint8_t T1_hash_shift;  // 32 to 61
	};

__forceinline uint64_t CalcHash(uint64_t key, uint8_t hash_shift)
	{
	// non-cryptographic hash
	// only use high 32 bits of result
	const uint64_t fib = 11400714819323198485;
	uint64_t hash = (key ^ (key >> 33)) * fib;
	hash = hash >> hash_shift;
	return hash;
	}


TableInfo* InitHashTable(uint32_t min_table_size, uint16_t value_size)
	// min_table_size should be at least 1.125x the expected # of elements
	{
	unsigned long log2_table_size;
	_BitScanReverse(&log2_table_size, (min_table_size - 1));
	log2_table_size++;
	log2_table_size = __max(log2_table_size, 3);
	uint64_t table_size = ((uint64_t)1) << log2_table_size;

	uint64_t element_size = value_size + KEY_BYTES;
	uint64_t header_element_size = element_size + 1;
	uint64_t T2_slot_count = (table_size >> 3);
	uint64_t array_size = (table_size + 2) * header_element_size + T2_slot_count * element_size;
	uintptr_t main_array = (uintptr_t)calloc(array_size, 1);

	uint32_t inserts_til_resize = ((T2_slot_count * 6) >> 3) + 1;  // T2 load factor = 0.75 (?)

	TableInfo info_scratch =
		{
		inserts_til_resize,
		0,
		(uint16_t)header_element_size,
		main_array + header_element_size,  // padded by 1 in each direction
		64 - log2_table_size
		};

	TableInfo* info_ptr = reinterpret_cast<TableInfo*>(malloc(sizeof(struct TableInfo)));
	memcpy(info_ptr, &info_scratch, sizeof(struct TableInfo));
	return info_ptr;
	}


uint8_t T2Insert(uintptr_t T1_source, uint64_t hash, TableInfo* info)
	{
	T1_source++;  // header not used
	uint64_t insert_key = *(reinterpret_cast<uint64_t*>(T1_source));
	uint32_t info_element_size_bytes = info->element_size_bytes;
	uint8_t info_T1_hash_shift = info->T1_hash_shift;
	uintptr_t info_T1_ptr = info->T1_ptr;

	uint8_t log2_T1_size = 64 - info_T1_hash_shift;
	uint32_t T1_size = ((uint32_t)1) << log2_T1_size;
	uintptr_t T2_start_ptr = info_T1_ptr + (info_element_size_bytes << log2_T1_size) + info_element_size_bytes;
	
	hash ^= hash << 2;  // reduce clustering (maybe += ?)
	uint32_t T2_stride = info_element_size_bytes - 1;  // no header bytes in T2
	uint64_t T2_start_index = hash & ((T1_size >> 3) - 1);  // bounds checking
	uintptr_t T2_pos = T2_start_ptr + (uintptr_t)(T2_start_index * T2_stride);
	uintptr_t T2_last_ptr = T2_start_ptr + (T2_stride << (log2_T1_size - 3));

	info->T2_inserts_til_resize--;  // TODO: automatic resizing
	

	uint32_t info_zero_key_location = info->zero_key_location;
	uintptr_t zero_key_ptr = 0;
	if (info_zero_key_location && insert_key != 0)
		zero_key_ptr = T2_start_ptr + (info_zero_key_location - 1) * T2_stride;

	uintptr_t T2_pos0 = T2_pos;  // saved for backwards scan

	uint32_t offset = 0;
	while (true)
		{
		uint64_t slot_key = *(reinterpret_cast<uint64_t*>(T2_pos));
		if (slot_key == 0 && zero_key_ptr != T2_pos)
			{
			memcpy((void*)T2_pos, (void*)T1_source, T2_stride);
			if (insert_key == 0)
				info->zero_key_location = offset + 1;
			
			return offset;
			}

		if (offset >= 62)
			break;
		offset++;

		if (T2_pos >= T2_last_ptr)
			T2_pos = T2_start_ptr;
		else
			T2_pos += T2_stride;
		}

	// after max offset, scan backwards from T2_pos0 for insertion (rare case)
	for (uint8_t i = 254; i != 0; i--)
		{
		if (T2_pos0 <= T2_start_ptr)
			T2_pos0 = T2_last_ptr;
		else
			T2_pos0 -= T2_stride;

		uint64_t slot_key = *(reinterpret_cast<uint64_t*>(T2_pos0));
		if (slot_key == 0 && zero_key_ptr != T2_pos0)
			{
			memcpy((void*)T2_pos0, (void*)T1_source, T2_stride);
			if (insert_key == 0)
				info->zero_key_location = ((T2_pos0 - T2_start_ptr) / T2_stride) + 1;

			return offset;  // max offset = 62
			}
		}

	return 0;  // failed insertion (T2 is too full)
	}


void InsertItem(uint64_t insert_key, uintptr_t value_ptr, TableInfo* info)  // bool replace_existing = true ?
	{
	uint8_t info_T1_hash_shift = info->T1_hash_shift;
	uint64_t hash = CalcHash(insert_key, info_T1_hash_shift);
	uint8_t last_header_match = 0;

	uintptr_t info_T1_ptr = info->T1_ptr;
	uint32_t info_element_size_bytes = info->element_size_bytes;
	size_t data_size = info_element_size_bytes - (KEY_BYTES + 1);

	uintptr_t pos0 = info_T1_ptr + (uintptr_t)(hash * info_element_size_bytes);
	uintptr_t key_pos = pos0 + 1;
	uint8_t header = *(reinterpret_cast<uint8_t*>(pos0));
	uint8_t offset = header & 3;
	uint8_t pos0_offset = offset;

	switch (offset)
		{
		case 0:
			{
			*(reinterpret_cast<uint8_t*>(pos0)) = (uint8_t)2;
			*(reinterpret_cast<uint64_t*>(key_pos)) = insert_key;
			memcpy((void*)(key_pos + KEY_BYTES), reinterpret_cast<void*>(value_ptr), data_size);
			return;
			}
		case 2:
			{
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(key_pos));
			if (slot_key == insert_key)
				{
				memcpy((void*)(key_pos + KEY_BYTES), (void*)value_ptr, data_size);
				return;
				}
			else
				last_header_match = header;
			}
			break;
		case 1:
			goto try_insert_left;
		default:
			break;
		}

	{
	uintptr_t pos1 = pos0 + info_element_size_bytes;
	header = *(reinterpret_cast<uint8_t*>(pos1));
	offset = header & 3;
	switch (offset)
		{
		case 0:  // maybe swap if pos0_offset==1 (only possible with element deletion, and that can be checked during deletion)
			{
			key_pos = pos1 + 1;
			*(reinterpret_cast<uint8_t*>(pos1)) = (uint8_t)3;
			*(reinterpret_cast<uint64_t*>(key_pos)) = insert_key;
			memcpy((void*)(key_pos + KEY_BYTES), reinterpret_cast<void*>(value_ptr), data_size);
			return;
			}
		case 3:
			{
			key_pos = pos1 + 1;
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(key_pos));
			if (slot_key == insert_key)
				{
				memcpy((void*)(key_pos + KEY_BYTES), (void*)value_ptr, data_size);
				return;
				}
			else
				last_header_match = header;
			}
			break;
		default:
			break;
		}
	}

	try_insert_left:
	{
	uintptr_t pos_n1 = pos0 - info_element_size_bytes;
	header = *(reinterpret_cast<uint8_t*>(pos_n1));
	offset = header & 3;
	switch (offset)
		{
		case 0:
			{
			key_pos = pos_n1 + 1;
			*(reinterpret_cast<uint8_t*>(pos_n1)) = (uint8_t)1;
			*(reinterpret_cast<uint64_t*>(key_pos)) = insert_key;
			memcpy((void*)(key_pos + KEY_BYTES), reinterpret_cast<void*>(value_ptr), data_size);
			return;
			}
		case 1:
			{
			key_pos = pos_n1 + 1;
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(key_pos));
			if (slot_key == insert_key)
				{
				memcpy((void*)(key_pos + KEY_BYTES), (void*)value_ptr, data_size);
				return;
				}
			else
				last_header_match = header;
			}
			break;
		default:
			break;
		}
	}


	// scan T2 for existing match before pushing
	if (last_header_match >= 4)
		{
		uint8_t log2_T1_size = 64 - info_T1_hash_shift;
		uint32_t T1_size = ((uint32_t)1) << log2_T1_size;
		uintptr_t T2_start_ptr = info_T1_ptr + (info_element_size_bytes << log2_T1_size) + info_element_size_bytes;
		uint32_t T2_offset = (last_header_match >> 2) - 1;

		uint32_t T2_hash = hash ^ (hash << 2);  // reduce clustering (maybe += ?)
		uint32_t T2_stride = info_element_size_bytes - 1;  // no header bytes in T2
		uint64_t T2_start_index = T2_hash + T2_offset;
		T2_start_index &= (T1_size >> 3) - 1;  // bounds checking
		uintptr_t T2_pos = T2_start_ptr + (uintptr_t)(T2_start_index * T2_stride);
		uint32_t T2_offset_original = T2_offset;

		if (insert_key == 0)
			{
			uint32_t info_zero_key_location = info->zero_key_location;
			if (info_zero_key_location != 0)
				{
				memcpy((void*)(T2_start_ptr + (T2_stride * (info_zero_key_location - 1)) + KEY_BYTES), reinterpret_cast<void*>(value_ptr), data_size);
				return;
				}
			else
				goto insert_pushing;
			}

		// reverse linear probing in T2, using T2_offset
		while (true)
			{
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(T2_pos));
			if (slot_key == insert_key)
				{
				memcpy((void*)(T2_pos + KEY_BYTES), reinterpret_cast<void*>(value_ptr), data_size);
				return;
				}

			if (T2_offset == 0)
				{
				if (T2_offset_original != 62)
					goto insert_pushing;
				T2_offset = 255;
				T2_offset_original = 0;
				}
			T2_offset--;

			T2_pos -= T2_stride;
			if (T2_pos < T2_start_ptr)
				T2_pos += T2_stride << (log2_T1_size - 3);
			}

		}


	insert_pushing:
	bool push_right = false;
	switch (pos0_offset)
		{
		case 2:
			if (hash & 1)
				{
				uintptr_t pos_n2 = pos0 - (info_element_size_bytes << 1);
				header = *(reinterpret_cast<uint8_t*>(pos_n2));
				push_right = header != 0;
				}
			else
				{
				uintptr_t pos2 = pos0 + (info_element_size_bytes << 1);
				header = *(reinterpret_cast<uint8_t*>(pos2));
				push_right = header == 0;
				}
			break;
		case 1:
			push_right = true;
			break;
		default: break;
		}

	uintptr_t header_scan_ptr = pos0;
	uint8_t header_old1 = 0;
	uint8_t header_old2 = 0;
	if (push_right)
		{
		while (true)
			{
			header = *(reinterpret_cast<uint8_t*>(header_scan_ptr));
			switch (header & 3)
				{
				case 3:
					{
					uint8_t new_offset = T2Insert(header_scan_ptr, hash-1, info);
					new_offset = __max((new_offset << 2) + 7, header);
					uintptr_t mid_header_ptr = header_scan_ptr - info_element_size_bytes;
					*(reinterpret_cast<uint8_t*>(mid_header_ptr)) = new_offset;
					if ((header_old2 & 3) == 1)
						{
						uintptr_t header_old2_ptr = mid_header_ptr - info_element_size_bytes;
						*(reinterpret_cast<uint8_t*>(header_old2_ptr)) = new_offset - 1;
						}
					}
				case 0:
					memmove((void*)(pos0 + info_element_size_bytes), (void*)pos0, (header_scan_ptr - pos0));
					key_pos = pos0 + 1;
					*(reinterpret_cast<uint8_t*>(pos0)) = (last_header_match & 252) | (uint8_t)2;
					*(reinterpret_cast<uint64_t*>(key_pos)) = insert_key;
					memcpy((void*)(key_pos + KEY_BYTES), reinterpret_cast<void*>(value_ptr), data_size);
					return;
				default:
					*(reinterpret_cast<uint8_t*>(header_scan_ptr)) = header + 1;
					header_scan_ptr += info_element_size_bytes;
					hash++;
					header_old2 = header_old1;
					header_old1 = header;
					break;
				}
			}
		}
	else
		{
		while (true)
			{
			header = *(reinterpret_cast<uint8_t*>(header_scan_ptr));
			switch (header & 3)
				{
				case 1:
					{
					uint8_t new_offset = T2Insert(header_scan_ptr, hash+1, info);
					new_offset = __max((new_offset << 2) + 5, header);
					uintptr_t mid_header_ptr = header_scan_ptr + info_element_size_bytes;
					*(reinterpret_cast<uint8_t*>(mid_header_ptr)) = new_offset;
					if ((header_old2 & 3) == 3)
						{
						uintptr_t header_old2_ptr = mid_header_ptr + info_element_size_bytes;
						*(reinterpret_cast<uint8_t*>(header_old2_ptr)) = new_offset + 1;
						}
					}
				case 0:
					memmove((void*)header_scan_ptr, (void*)(header_scan_ptr + info_element_size_bytes), (pos0 - header_scan_ptr));
					key_pos = pos0 + 1;
					*(reinterpret_cast<uint8_t*>(pos0)) = (last_header_match & 252) | (uint8_t)2;
					*(reinterpret_cast<uint64_t*>(key_pos)) = insert_key;
					memcpy((void*)(key_pos + KEY_BYTES), reinterpret_cast<void*>(value_ptr), data_size);
					return;
				default:
					*(reinterpret_cast<uint8_t*>(header_scan_ptr)) = header - 1;
					header_scan_ptr -= info_element_size_bytes;
					hash--;
					header_old2 = header_old1;
					header_old1 = header;
					break;
				}
			}
		}

	return;
	}


uintptr_t FindItem(uint64_t search_key, TableInfo* info)
	{
	uint8_t info_T1_hash_shift = info->T1_hash_shift;
	uint64_t hash = CalcHash(search_key, info_T1_hash_shift);
	uint8_t last_header_match = 0;

	uintptr_t info_T1_ptr = info->T1_ptr;
	uint32_t info_element_size_bytes = info->element_size_bytes;
	uintptr_t pos0 = info_T1_ptr + (uintptr_t)(hash * info_element_size_bytes);
	uint8_t header = *(reinterpret_cast<uint8_t*>(pos0));
	uint8_t offset = header & 3;
	switch (offset)
		{
		case 2:
			{
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(pos0 + 1));
			if (slot_key == search_key)
				return (pos0 + KEY_BYTES + 1);
			else
				last_header_match = header;
			}
			break;
		case 0:
			return 0;
		case 1:
			goto check_left;
		default:
			break;
		}

	{
	uintptr_t pos1 = pos0 + info_element_size_bytes;
	header = *(reinterpret_cast<uint8_t*>(pos1));
	offset = header & 3;
	switch (offset)
		{
		case 3:
			{
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(pos1 + 1));
			if (slot_key == search_key)
				return (pos1 + KEY_BYTES + 1);
			else
				last_header_match = header;
			}
			break;
		case 0:
			return 0;
		default:
			break;
		}
	}

	check_left:
	{
	uintptr_t pos_n1 = pos0 - info_element_size_bytes;
	header = *(reinterpret_cast<uint8_t*>(pos_n1));
	offset = header & 3;
	switch (offset)
		{
		case 1:
			{
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(pos_n1 + 1));
			if (slot_key == search_key)
				return (pos_n1 + KEY_BYTES + 1);
			else
				last_header_match = header;
			}
			break;
		case 0:
			return 0;
		default:
			break;
		}
	}

	if (last_header_match >= 4)  // check T2
		{
		uint8_t log2_T1_size = 64 - info_T1_hash_shift;
		uint32_t T1_size = ((uint32_t)1) << log2_T1_size;
		uintptr_t T2_start_ptr = info_T1_ptr + (info_element_size_bytes << log2_T1_size) + info_element_size_bytes;
		uint32_t T2_offset = (last_header_match >> 2) - 1;
		
		hash ^= hash << 2;  // reduce clustering (maybe += ?)
		uint32_t T2_stride = info_element_size_bytes - 1;  // no header bytes in T2
		uint64_t T2_start_index = hash + T2_offset;
		T2_start_index &= (T1_size >> 3) - 1;  // bounds checking
		uintptr_t T2_pos = T2_start_ptr + (uintptr_t)(T2_start_index * T2_stride);
		uint32_t T2_offset_original = T2_offset;

		if (search_key == 0)
			{
			uint32_t info_zero_key_location = info->zero_key_location;
			if (info_zero_key_location != 0)
				return T2_start_ptr + (T2_stride * (info_zero_key_location - 1)) + KEY_BYTES;
			else
				return 0;
			}

		// reverse linear probing in T2, using T2_offset
		while (true)
			{
			uint64_t slot_key = *(reinterpret_cast<uint64_t*>(T2_pos));
			if (slot_key == search_key)
				return (T2_pos + KEY_BYTES);

			if (T2_offset == 0)
				{
				if (T2_offset_original != 62)
					return 0;
				T2_offset = 255;
				T2_offset_original = 0;
				}
			T2_offset--;

			T2_pos -= T2_stride;
			if (T2_pos < T2_start_ptr)
				T2_pos += T2_stride << (log2_T1_size - 3);
			}

		return 0;
		}

	return 0;
	}


int main()
{
	TableInfo* info = InitHashTable(256, 8);
	for (int i = 0; i < 228; i++)
		{
		uint64_t my_data = i * 100;
		InsertItem(i*i*3, reinterpret_cast<uintptr_t>((&my_data)), info);
		}
	for (int i = 0; i < 230; i++)
		{
		uintptr_t data_ptr = FindItem(i*i*3, info);
		if (data_ptr == 0)
			std::cout << "not_found\n";
		}
}


